// Includes
#include <PetscSolver3DHandler.h>
#include <MathUtils.h>
#include <Constants.h>

namespace xolotlSolver {

void PetscSolver3DHandler::createSolverContext(DM &da) {
	PetscErrorCode ierr;
	// Recompute Ids and network size and redefine the connectivities
	network.reinitializeConnectivities();

	// Degrees of freedom is the total number of clusters in the network
	const int dof = network.getDOF();

	// Set the position of the surface
	// Loop on Y
	for (int j = 0; j < nY; j++) {
		// Create a one dimensional vector to store the surface indices
		// for a given Y position
		std::vector<int> tempPosition;

		// Loop on Z
		for (int k = 0; k < nZ; k++) {
			tempPosition.push_back(0);
			if (movingSurface)
				tempPosition[k] = (int) (nX * portion / 100.0);
		}

		// Add tempPosition to the surfacePosition
		surfacePosition.push_back(tempPosition);
	}

	// Generate the grid in the x direction
	generateGrid(nX, hX, surfacePosition[0][0]);

	// Now that the grid was generated, we can update the surface position
	// if we are using a restart file
	if (not networkName.empty() and movingSurface) {
		xolotlCore::XFile xfile(networkName);
		auto concGroup =
				xfile.getGroup<xolotlCore::XFile::ConcentrationGroup>();
		if (concGroup and concGroup->hasTimesteps()) {

			auto tsGroup = concGroup->getLastTimestepGroup();
			assert(tsGroup);

			auto surfaceIndices = tsGroup->readSurface3D();

			// Set the actual surface positions
			for (int i = 0; i < surfaceIndices.size(); i++) {
				for (int j = 0; j < surfaceIndices[0].size(); j++) {
					surfacePosition[i][j] = surfaceIndices[i][j];
				}
			}

		}
	}

	// Prints the grid on one process
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);
	if (procId == 0) {
		for (int i = 1; i < grid.size() - 1; i++) {
			std::cout << grid[i] - grid[surfacePosition[0][0] + 1] << " ";
		}
		std::cout << std::endl;
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Create distributed array (DMDA) to manage parallel grid and vectors
	 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	if (isMirror) {
		ierr = DMDACreate3d(xolotlComm, DM_BOUNDARY_MIRROR,
				DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
				nX, nY, nZ, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, 1,
				NULL,
				NULL, NULL, &da);
		checkPetscError(ierr, "PetscSolver3DHandler::createSolverContext: "
				"DMDACreate3d failed.");
	} else {
		ierr = DMDACreate3d(xolotlComm, DM_BOUNDARY_PERIODIC,
				DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
				nX, nY, nZ, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, 1,
				NULL,
				NULL, NULL, &da);
		checkPetscError(ierr, "PetscSolver3DHandler::createSolverContext: "
				"DMDACreate3d failed.");
	}
	ierr = DMSetFromOptions(da);
	checkPetscError(ierr,
			"PetscSolver3DHandler::createSolverContext: DMSetFromOptions failed.");
	ierr = DMSetUp(da);
	checkPetscError(ierr,
			"PetscSolver3DHandler::createSolverContext: DMSetUp failed.");

	// Initialize the surface of the first advection handler corresponding to the
	// advection toward the surface (or a dummy one if it is deactivated)
	advectionHandlers[0]->setLocation(
			grid[surfacePosition[0][0] + 1] - grid[1]);

	// Set the size of the partial derivatives vectors
	reactingPartialsForCluster.resize(dof, 0.0);

	/*  The only spatial coupling in the Jacobian is due to diffusion.
	 *  The ofill (thought of as a dof by dof 2d (row-oriented) array represents
	 *  the nonzero coupling between degrees of freedom at one point with degrees
	 *  of freedom on the adjacent point to the left or right. A 1 at i,j in the
	 *  ofill array indicates that the degree of freedom i at a point is coupled
	 *  to degree of freedom j at the adjacent point.
	 *  In this case ofill has only a few diagonal entries since the only spatial
	 *  coupling is regular diffusion.
	 */
	xolotlCore::IReactionNetwork::SparseFillMap ofill;
	xolotlCore::IReactionNetwork::SparseFillMap dfill;

	// Initialize the temperature handler
	temperatureHandler->initializeTemperature(network, ofill, dfill);

	// Fill ofill, the matrix of "off-diagonal" elements that represents diffusion
	diffusionHandler->initializeOFill(network, ofill);
	// Loop on the advection handlers to account the other "off-diagonal" elements
	for (int i = 0; i < advectionHandlers.size(); i++) {
		advectionHandlers[i]->initialize(network, ofill);
	}

	// Get the local boundaries
	PetscInt xs, xm, ys, ym, zs, zm;
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	checkPetscError(ierr, "PetscSolver3DHandler::createSolverContext: "
			"DMDAGetCorners failed.");
	// Set it in the handler
	setLocalCoordinates(xs, xm, ys, ym, zs, zm);

	// Initialize the modified trap-mutation handler because it adds connectivity
	mutationHandler->initialize(network, localXM, localYM, localZM);
	mutationHandler->initializeIndex3D(surfacePosition, network,
			advectionHandlers, grid, localXM, localXS, localYM, hY, localYS,
			localZM, hZ, localZS);

	// Initialize the re-solution handler here
	// because it adds connectivity
	resolutionHandler->initialize(network, electronicStoppingPower);

	// Get the diagonal fill
	network.getDiagonalFill(dfill);

	// Load up the block fills
	auto dfillsparse = ConvertToPetscSparseFillMap(dof, dfill);
	auto ofillsparse = ConvertToPetscSparseFillMap(dof, ofill);
	ierr = DMDASetBlockFillsSparse(da, dfillsparse.data(), ofillsparse.data());
	checkPetscError(ierr, "PetscSolver3DHandler::createSolverContext: "
			"DMDASetBlockFills failed.");

	// Initialize the arrays for the reaction partial derivatives
	reactionSize.resize(dof);
	reactionStartingIdx.resize(dof);
	auto nPartials = network.initPartialsSizes(reactionSize,
			reactionStartingIdx);

	reactionIndices.resize(nPartials);
	network.initPartialsIndices(reactionSize, reactionStartingIdx,
			reactionIndices);
	reactionVals.resize(nPartials);

	return;
}

void PetscSolver3DHandler::initializeConcentration(DM &da, Vec &C) {
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar ****concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::initializeConcentration: "
			"DMDAVecGetArrayDOF failed.");

	// Initialize the last temperature at each grid point on this process
	for (int i = 0; i < localXM + 2; i++) {
		lastTemperature.push_back(0.0);
	}
	network.addGridPoints(localXM + 2);

	// Get the last time step written in the HDF5 file
	bool hasConcentrations = false;
	std::unique_ptr<xolotlCore::XFile> xfile;
	std::unique_ptr<xolotlCore::XFile::ConcentrationGroup> concGroup;
	if (not networkName.empty()) {

		xfile.reset(new xolotlCore::XFile(networkName));
		concGroup = xfile->getGroup<xolotlCore::XFile::ConcentrationGroup>();
		hasConcentrations = (concGroup and concGroup->hasTimesteps());
	}

	// Give the surface position to the temperature handler
	temperatureHandler->updateSurfacePosition(surfacePosition[0][0]);

	// Initialize the flux handler
	fluxHandler->initializeFluxHandler(network, surfacePosition[0][0], grid);

	// Initialize the grid for the diffusion
	diffusionHandler->initializeDiffusionGrid(advectionHandlers, grid, localXM,
			localXS, localYM, hY, localYS, localZM, hZ, localZS);

	// Initialize the grid for the advection
	advectionHandlers[0]->initializeAdvectionGrid(advectionHandlers, grid,
			localXM, localXS, localYM, hY, localYS, localZM, hZ, localZS);

	// Pointer for the concentration vector at a specific grid point
	PetscScalar *concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	const int dof = network.getDOF();

	// Get the single vacancy ID
	auto singleVacancyCluster = network.get(xolotlCore::Species::V, 1);
	int vacancyIndex = -1;
	if (singleVacancyCluster)
		vacancyIndex = singleVacancyCluster->getId() - 1;

	// Loop on all the grid points
	for (int k = localZS; k < localZS + localZM; k++) {
		for (int j = localYS; j < localYS + localYM; j++) {
			for (int i = localXS; i < localXS + localXM; i++) {
				concOffset = concentrations[k][j][i];

				// Loop on all the clusters to initialize at 0.0
				for (int n = 0; n < dof - 1; n++) {
					concOffset[n] = 0.0;
				}

				// Temperature
				xolotlCore::NDPoint<3> gridPosition { ((grid[i] + grid[i + 1])
						/ 2.0 - grid[surfacePosition[j][k] + 1])
						/ (grid[grid.size() - 1]
								- grid[surfacePosition[j][k] + 1]), 0.0, 0.0 };
				concOffset[dof - 1] = temperatureHandler->getTemperature(
						gridPosition, 0.0);

				// Initialize the vacancy concentration
				if (i >= surfacePosition[j][k] + leftOffset && vacancyIndex > 0
						&& !hasConcentrations && i < nX - rightOffset
						&& j >= bottomOffset && j < nY - topOffset
						&& k >= frontOffset && k < nZ - backOffset) {
					concOffset[vacancyIndex] = initialVConc;
				}
			}
		}
	}

	// If the concentration must be set from the HDF5 file
	if (hasConcentrations) {

		assert(concGroup);
		auto tsGroup = concGroup->getLastTimestepGroup();
		assert(tsGroup);

		// Loop on the full grid
		for (int k = 0; k < nZ; k++) {
			for (int j = 0; j < nY; j++) {
				for (int i = 0; i < nX; i++) {
					// Read the concentrations from the HDF5 file
					auto concVector = tsGroup->readGridPoint(i, j, k);

					// Change the concentration only if we are on the locally
					// owned part of the grid
					if (i >= localXS && i < localXS + localXM && j >= localYS
							&& j < localYS + localYM && k >= localZS
							&& k < localZS + localZM) {
						concOffset = concentrations[k][j][i];
						// Loop on the concVector size
						for (unsigned int l = 0; l < concVector.size(); l++) {
							concOffset[(int) concVector.at(l).at(0)] =
									concVector.at(l).at(1);
						}
						// Set the temperature in the network
						double temp = concVector.at(concVector.size() - 1).at(
								1);
						network.setTemperature(temp, i - localXS);
						// Update the modified trap-mutation rate
						// that depends on the network reaction rates
						mutationHandler->updateTrapMutationRate(network);
						lastTemperature[i - localXS] = temp;
					}
				}
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::initializeConcentration: "
			"DMDAVecRestoreArrayDOF failed.");

	// Set the rate for re-solution
	resolutionHandler->updateReSolutionRate(fluxHandler->getFluxAmplitude());

	return;
}

void PetscSolver3DHandler::initGBLocation(DM &da, Vec &C) {
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar ****concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::initGBLocation: "
			"DMDAVecGetArrayDOF failed.");

	// Pointer for the concentration vector at a specific grid point
	PetscScalar *concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	// + the super clusters
	const int dof = network.getDOF();

	// Loop on the GB
	for (auto const& pair : gbVector) {
		// Get the coordinate of the point
		int xi = std::get<0>(pair);
		int yj = std::get<1>(pair);
		int zk = std::get<2>(pair);
		// Check if we are on the right process
		if (xi >= localXS && xi < localXS + localXM && yj >= localYS
				&& yj < localYS + localYM && zk >= localZS
				&& zk < localZS + localZM) {
			// Get the local concentration
			concOffset = concentrations[zk][yj][xi];

			// Update the concentration in the network
			network.updateConcentrationsFromArray(concOffset);

			// Add this Xe concentration to the Xe rate
			setLocalXeRate(network.getTotalAtomConcentration(), xi - localXS, yj - localYS, zk - localZS);

			// Loop on all the clusters to initialize at 0.0
			for (int n = 0; n < dof - 1; n++) {
				concOffset[n] = 0.0;
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::initGBLocation: "
			"DMDAVecRestoreArrayDOF failed.");

	return;
}

std::vector<std::vector<std::vector<std::vector<std::pair<int, double> > > > > PetscSolver3DHandler::getConcVector(
		DM &da, Vec &C) {

	// Initial declaration
	PetscErrorCode ierr;
	const double *gridPointSolution = nullptr;

	// Pointer for the concentration vector
	PetscScalar ****concentrations = nullptr;
	ierr = DMDAVecGetArrayDOFRead(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::getConcVector: "
			"DMDAVecGetArrayDOFRead failed.");

	// Get the network and dof
	auto& network = getNetwork();
	const int dof = network.getDOF();

	// Create the vector for the concentrations
	std::vector<std::vector<std::vector<std::vector<std::pair<int, double> > > > > toReturn;

	// Loop on the grid points
	for (auto k = 0; k < localZM; ++k) {
		std::vector<std::vector<std::vector<std::pair<int, double> > > > tempTempTempVector;
		for (auto j = 0; j < localYM; ++j) {
			std::vector<std::vector<std::pair<int, double> > > tempTempVector;
			for (auto i = 0; i < localXM; ++i) {
				gridPointSolution =
						concentrations[localZS + k][localYS + j][localXS + i];

				// Create the temporary vector for this grid point
				std::vector<std::pair<int, double> > tempVector;
				for (auto l = 0; l < dof; ++l) {
					if (std::fabs(gridPointSolution[l]) > 1.0e-16) {
						tempVector.push_back(
								std::make_pair(l, gridPointSolution[l]));
					}
				}
				tempTempVector.push_back(tempVector);
			}
			tempTempTempVector.push_back(tempTempVector);
		}
		toReturn.push_back(tempTempTempVector);
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::getConcVector: "
			"DMDAVecRestoreArrayDOFRead failed.");

	return toReturn;
}

void PetscSolver3DHandler::setConcVector(DM &da, Vec &C,
		std::vector<
				std::vector<std::vector<std::vector<std::pair<int, double> > > > > & concVector) {
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar *gridPointSolution = nullptr;
	PetscScalar ****concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMDAVecGetArrayDOF failed.");

	// Loop on the grid points
	for (auto k = 0; k < localZM; ++k) {
		for (auto j = 0; j < localYM; ++j) {
			for (auto i = 0; i < localXM; ++i) {
				gridPointSolution =
						concentrations[localZS + k][localYS + j][localXS + i];

				// Loop on the given vector
				for (int l = 0; l < concVector[k][j][i].size(); l++) {
					gridPointSolution[concVector[k][j][i][l].first] =
							concVector[k][j][i][l].second;
				}
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMDAVecRestoreArrayDOF failed.");

	// Get the complete data array, including ghost cells to set the temperature at the ghost points
	Vec localSolution;
	ierr = DMGetLocalVector(da, &localSolution);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMGetLocalVector failed.");
	ierr = DMGlobalToLocalBegin(da, C, INSERT_VALUES, localSolution);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMGlobalToLocalBegin failed.");
	ierr = DMGlobalToLocalEnd(da, C, INSERT_VALUES, localSolution);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMGlobalToLocalEnd failed.");
	// Get the array of concentration
	ierr = DMDAVecGetArrayDOFRead(da, localSolution, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMDAVecGetArrayDOFRead failed.");

	// Getthe DOF of the network
	const int dof = network.getDOF();

	// Loop on the grid points
	for (auto k = 0; k < localZM; ++k) {
		for (auto j = 0; j < localYM; ++j) {
			for (auto i = -1; i <= localXM; ++i) {
				gridPointSolution =
						concentrations[localZS + k][localYS + j][localXS + i];

				// Set the temperature in the network
				double temp = gridPointSolution[dof - 1];
				network.setTemperature(temp, i + 1);
				// Update the modified trap-mutation rate
				// that depends on the network reaction rates
				mutationHandler->updateTrapMutationRate(network);
				lastTemperature[i + 1] = temp;
			}
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, localSolution, &concentrations);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMDAVecRestoreArrayDOFRead failed.");
	ierr = DMRestoreLocalVector(da, &localSolution);
	checkPetscError(ierr, "PetscSolver3DHandler::setConcVector: "
			"DMRestoreLocalVector failed.");

	return;
}

void PetscSolver3DHandler::updateConcentration(TS &ts, Vec &localC, Vec &F,
		PetscReal ftime) {
	PetscErrorCode ierr;

	// Get the local data vector from PETSc
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(ierr, "PetscSolver3DHandler::updateConcentration: "
			"TSGetDM failed.");

	// Pointers to the PETSc arrays that start at the beginning (xs, ys, zs) of the
	// local array!
	PetscScalar ****concs = nullptr, ****updatedConcs = nullptr;
	// Get pointers to vector data
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver3DHandler::updateConcentration: "
			"DMDAVecGetArrayDOFRead (localC) failed.");
	ierr = DMDAVecGetArrayDOF(da, F, &updatedConcs);
	checkPetscError(ierr, "PetscSolver3DHandler::updateConcentration: "
			"DMDAVecGetArrayDOF (F) failed.");

	// The following pointers are set to the first position in the conc or
	// updatedConc arrays that correspond to the beginning of the data for the
	// current grid point. They are accessed just like regular arrays.
	PetscScalar *concOffset = nullptr, *updatedConcOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	const int dof = network.getDOF();

	// Set some step size variable
	double sy = 1.0 / (hY * hY);
	double sz = 1.0 / (hZ * hZ);

	// Declarations for variables used in the loop
	double **concVector = new double*[7];
	xolotlCore::NDPoint<3> gridPosition { 0.0, 0.0, 0.0 };
	std::vector<double> incidentFluxVector;
	double atomConc = 0.0, totalAtomConc = 0.0;

	// Loop over grid points
	for (int zk = frontOffset; zk < nZ - backOffset; zk++) {
		for (int yj = bottomOffset; yj < nY - topOffset; yj++) {

			// Computing the trapped atom concentration is only needed for the attenuation
			if (useAttenuation) {
				// Compute the total concentration of atoms contained in bubbles
				atomConc = 0.0;

				// Loop over grid points
				for (int xi = surfacePosition[yj][zk] + leftOffset;
						xi < nX - rightOffset; xi++) {
					// We are only interested in the helium near the surface
					if ((grid[xi] + grid[xi + 1]) / 2.0
							- grid[surfacePosition[yj][zk] + 1] > 2.0)
						continue;

					// Check if we are on the right processor
					if (xi >= localXS && xi < localXS + localXM && yj >= localYS
							&& yj < localYS + localYM && zk >= localZS
							&& zk < localZS + localZM) {
						// Get the concentrations at this grid point
						concOffset = concs[zk][yj][xi];
						// Copy data into the PSIClusterReactionNetwork
						network.updateConcentrationsFromArray(concOffset);

						// Sum the total atom concentration
						atomConc += network.getTotalTrappedAtomConcentration()
								* (grid[xi + 1] - grid[xi]);
					}
				}

				// Share the concentration with all the processes
				totalAtomConc = 0.0;
				auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
				MPI_Allreduce(&atomConc, &totalAtomConc, 1, MPI_DOUBLE, MPI_SUM,
						xolotlComm);

				// Set the disappearing rate in the modified TM handler
				mutationHandler->updateDisappearingRate(totalAtomConc);
			}

			// Skip if we are not on the right process
			if (yj < localYS || yj >= localYS + localYM || zk < localZS
					|| zk >= localZS + localZM)
				continue;

			// Set the grid position
			gridPosition[1] = yj * hY;
			gridPosition[2] = zk * hZ;

			// Initialize the flux, advection, and temperature handlers which depend
			// on the surface position at Y
			fluxHandler->initializeFluxHandler(network, surfacePosition[yj][zk],
					grid);
			advectionHandlers[0]->setLocation(
					grid[surfacePosition[yj][zk] + 1] - grid[1]);
			temperatureHandler->updateSurfacePosition(surfacePosition[yj][zk]);

			for (int xi = localXS; xi < localXS + localXM; xi++) {
				// Compute the old and new array offsets
				concOffset = concs[zk][yj][xi];
				updatedConcOffset = updatedConcs[zk][yj][xi];

				// Fill the concVector with the pointer to the middle, left,
				// right, bottom, top, front, and back grid points
				concVector[0] = concOffset;				// middle
				concVector[1] = concs[zk][yj][xi - 1];				// left
				concVector[2] = concs[zk][yj][xi + 1];				// right
				concVector[3] = concs[zk][yj - 1][xi];			// bottom
				concVector[4] = concs[zk][yj + 1][xi];				// top
				concVector[5] = concs[zk - 1][yj][xi];				// front
				concVector[6] = concs[zk + 1][yj][xi];				// back

				// Compute the left and right hx
				double hxLeft = 0.0, hxRight = 0.0;
				if (xi - 1 >= 0 && xi + 2 < nX + 2) {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				} else if (xi - 1 < 0) {
					hxLeft = grid[xi + 1] - grid[xi];
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				} else {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = grid[xi + 1] - grid[xi];
				}

				// Heat condition
				if (xi == surfacePosition[yj][zk]) {
					temperatureHandler->computeTemperature(concVector,
							updatedConcOffset, hxLeft, hxRight, xi, sy, yj, sz,
							zk);
				}

				// Boundary conditions
				// Everything to the left of the surface is empty
				if (xi < surfacePosition[yj][zk] + leftOffset
						|| xi > nX - 1 - rightOffset || yj < bottomOffset
						|| yj > nY - 1 - topOffset || zk < frontOffset
						|| zk > nZ - 1 - backOffset) {
					continue;
				}
				// Free surface GB
				bool skip = false;
				for (auto &pair : gbVector) {
					if (xi == std::get<0>(pair) && yj == std::get<1>(pair)
							&& zk == std::get<2>(pair)) {
						skip = true;
						break;
					}
				}
				if (skip)
					continue;

				// Update the network if the temperature changed
				// left
				double temperature = concs[zk][yj][xi - 1][dof - 1];
				if (std::fabs(lastTemperature[xi - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi - localXS);
					lastTemperature[xi - localXS] = temperature;
				}
				// right
				temperature = concs[zk][yj][xi + 1][dof - 1];
				if (std::fabs(lastTemperature[xi + 2 - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi + 2 - localXS);
					lastTemperature[xi + 2 - localXS] = temperature;
				}

				// Set the grid fraction
				gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0
						- grid[surfacePosition[yj][zk] + 1])
						/ (grid[grid.size() - 1]
								- grid[surfacePosition[yj][zk] + 1]);

				// Get the temperature from the temperature handler
				temperatureHandler->setTemperature(concOffset);
				temperature = temperatureHandler->getTemperature(gridPosition,
						ftime);
				// middle
				if (std::fabs(lastTemperature[xi + 1 - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi + 1 - localXS);
					// Update the modified trap-mutation rate
					// that depends on the network reaction rates
					mutationHandler->updateTrapMutationRate(network);
					lastTemperature[xi + 1 - localXS] = temperature;
				}

				// Copy data into the ReactionNetwork so that it can
				// compute the fluxes properly. The network is only used to compute the
				// fluxes and hold the state data from the last time step. I'm reusing
				// it because it cuts down on memory significantly (about 400MB per
				// grid point) at the expense of being a little tricky to comprehend.
				network.updateConcentrationsFromArray(concOffset);

				// ----- Account for flux of incoming particles -----
				fluxHandler->computeIncidentFlux(ftime, updatedConcOffset, xi,
						surfacePosition[yj][zk]);

				// ---- Compute the temperature over the locally owned part of the grid -----
				temperatureHandler->computeTemperature(concVector,
						updatedConcOffset, hxLeft, hxRight, xi, sy, yj, sz, zk);

				// ---- Compute diffusion over the locally owned part of the grid -----
				diffusionHandler->computeDiffusion(network, concVector,
						updatedConcOffset, hxLeft, hxRight, xi - localXS, sy,
						yj - localYS, sz, zk - localZS);

				// ---- Compute advection over the locally owned part of the grid -----
				// Set the grid position
				gridPosition[0] = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
				for (int i = 0; i < advectionHandlers.size(); i++) {
					advectionHandlers[i]->computeAdvection(network,
							gridPosition, concVector, updatedConcOffset, hxLeft,
							hxRight, xi - localXS, hY, yj - localYS, hZ,
							zk - localZS);
				}

				// ----- Compute the modified trap-mutation over the locally owned part of the grid -----
				mutationHandler->computeTrapMutation(network, concOffset,
						updatedConcOffset, xi - localXS, yj - localYS,
						zk - localZS);

				// ----- Compute the re-solution over the locally owned part of the grid -----
				resolutionHandler->computeReSolution(network, concOffset,
						updatedConcOffset, xi, localXS, yj, zk);

				// ----- Compute the reaction fluxes over the locally owned part of the grid -----
				network.computeAllFluxes(updatedConcOffset, xi + 1 - localXS);
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver3DHandler::updateConcentration: "
			"DMDAVecRestoreArrayDOFRead (localC) failed.");
	ierr = DMDAVecRestoreArrayDOF(da, F, &updatedConcs);
	checkPetscError(ierr, "PetscSolver3DHandler::updateConcentration: "
			"DMDAVecRestoreArrayDOF (F) failed.");

	// Clear memory
	delete[] concVector;

	return;
}

void PetscSolver3DHandler::computeOffDiagonalJacobian(TS &ts, Vec &localC,
		Mat &J, PetscReal ftime) {
	PetscErrorCode ierr;

	// Get the distributed array
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(ierr, "PetscSolver3DHandler::computeOffDiagonalJacobian: "
			"TSGetDM failed.");

	// Setup some step size variables
	double sy = 1.0 / (hY * hY);
	double sz = 1.0 / (hZ * hZ);

	// Pointers to the PETSc arrays that start at the beginning (xs) of the
	// local array!
	PetscScalar ****concs = nullptr;
	// Get pointers to vector data
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver3DHandler::computeOffDiagonalJacobian: "
			"DMDAVecGetArrayDOFRead (localC) failed.");

	// Pointer to the concentrations at a given grid point
	PetscScalar *concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	const int dof = network.getDOF();

	// Get the total number of diffusing clusters
	const int nDiff = max(diffusionHandler->getNumberOfDiffusing(), 0);

	// Get the total number of advecting clusters
	int nAdvec = 0;
	for (int l = 0; l < advectionHandlers.size(); l++) {
		int n = advectionHandlers[l]->getNumberOfAdvecting();
		if (n > nAdvec)
			nAdvec = n;
	}

	// Arguments for MatSetValuesStencil called below
	MatStencil row, cols[7];
	PetscScalar tempVals[7];
	PetscInt tempIndices[1];
	PetscScalar diffVals[7 * nDiff];
	PetscInt diffIndices[nDiff];
	PetscScalar advecVals[2 * nAdvec];
	PetscInt advecIndices[nAdvec];
	xolotlCore::NDPoint<3> gridPosition { 0.0, 0.0, 0.0 };

	/*
	 Loop over grid points computing Jacobian terms for diffusion and advection
	 at each grid point
	 */
	for (int zk = localZS; zk < localZS + localZM; zk++) {
		// Set the grid position
		gridPosition[2] = zk * hZ;
		for (int yj = localYS; yj < localYS + localYM; yj++) {
			// Set the grid position
			gridPosition[1] = yj * hY;

			// Initialize the advection and temperature handlers which depend
			// on the surface position at Y
			advectionHandlers[0]->setLocation(
					grid[surfacePosition[yj][zk] + 1] - grid[1]);
			temperatureHandler->updateSurfacePosition(surfacePosition[yj][zk]);

			for (int xi = localXS; xi < localXS + localXM; xi++) {
				// Compute the left and right hx
				double hxLeft = 0.0, hxRight = 0.0;
				if (xi - 1 >= 0 && xi < nX) {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				} else if (xi - 1 < 0) {
					hxLeft = grid[xi + 1] - grid[xi];
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				} else {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = grid[xi + 1] - grid[xi];
				}

				// Heat condition
				if (xi == surfacePosition[yj][zk]) {
					// Get the partial derivatives for the temperature
					temperatureHandler->computePartialsForTemperature(tempVals,
							tempIndices, hxLeft, hxRight, xi, sy, yj, sz, zk);

					// Set grid coordinate and component number for the row
					row.i = xi;
					row.j = yj;
					row.k = zk;
					row.c = tempIndices[0];

					// Set grid coordinates and component numbers for the columns
					// corresponding to the middle, left, and right grid points
					cols[0].i = xi;					// middle
					cols[0].j = yj;
					cols[0].k = zk;
					cols[0].c = tempIndices[0];
					cols[1].i = xi - 1; // left
					cols[1].j = yj;
					cols[1].k = zk;
					cols[1].c = tempIndices[0];
					cols[2].i = xi + 1; // right
					cols[2].j = yj;
					cols[2].k = zk;
					cols[2].c = tempIndices[0];
					cols[3].i = xi; // bottom
					cols[3].j = yj - 1;
					cols[3].k = zk;
					cols[3].c = tempIndices[0];
					cols[4].i = xi; // top
					cols[4].j = yj + 1;
					cols[4].k = zk;
					cols[4].c = tempIndices[0];
					cols[5].i = xi; // front
					cols[5].j = yj;
					cols[5].k = zk - 1;
					cols[5].c = tempIndices[0];
					cols[6].i = xi; // back
					cols[6].j = yj;
					cols[6].k = zk + 1;
					cols[6].c = tempIndices[0];

					ierr = MatSetValuesStencil(J, 1, &row, 7, cols, tempVals,
							ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeOffDiagonalJacobian: "
									"MatSetValuesStencil (temperature) failed.");

				}

				// Boundary conditions
				// Everything to the left of the surface is empty
				if (xi < surfacePosition[yj][zk] + leftOffset
						|| xi > nX - 1 - rightOffset || yj < bottomOffset
						|| yj > nY - 1 - topOffset || zk < frontOffset
						|| zk > nZ - 1 - backOffset)
					continue;
				// Free surface GB
				bool skip = false;
				for (auto &pair : gbVector) {
					if (xi == std::get<0>(pair) && yj == std::get<1>(pair)
							&& zk == std::get<2>(pair)) {
						skip = true;
						break;
					}
				}
				if (skip)
					continue;

				// Update the network if the temperature changed
				// left
				double temperature = concs[zk][yj][xi - 1][dof - 1];
				if (std::fabs(lastTemperature[xi - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi - localXS);
					lastTemperature[xi - localXS] = temperature;
				}
				// right
				temperature = concs[zk][yj][xi + 1][dof - 1];
				if (std::fabs(lastTemperature[xi + 2 - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi + 2 - localXS);
					lastTemperature[xi + 2 - localXS] = temperature;
				}

				// Set the grid fraction
				gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0
						- grid[surfacePosition[yj][zk] + 1])
						/ (grid[grid.size() - 1]
								- grid[surfacePosition[yj][zk] + 1]);

				// Get the temperature from the temperature handler
				concOffset = concs[zk][yj][xi];
				temperatureHandler->setTemperature(concOffset);
				temperature = temperatureHandler->getTemperature(gridPosition,
						ftime);
				// middle
				if (std::fabs(lastTemperature[xi + 1 - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi + 1 - localXS);
					lastTemperature[xi + 1 - localXS] = temperature;
				}

				// Get the partial derivatives for the temperature
				temperatureHandler->computePartialsForTemperature(tempVals,
						tempIndices, hxLeft, hxRight, xi, sy, yj, sz, zk);

				// Set grid coordinate and component number for the row
				row.i = xi;
				row.j = yj;
				row.k = zk;
				row.c = tempIndices[0];

				// Set grid coordinates and component numbers for the columns
				// corresponding to the middle, left, and right grid points
				cols[0].i = xi;				// middle
				cols[0].j = yj;
				cols[0].k = zk;
				cols[0].c = tempIndices[0];
				cols[1].i = xi - 1; // left
				cols[1].j = yj;
				cols[1].k = zk;
				cols[1].c = tempIndices[0];
				cols[2].i = xi + 1; // right
				cols[2].j = yj;
				cols[2].k = zk;
				cols[2].c = tempIndices[0];
				cols[3].i = xi; // bottom
				cols[3].j = yj - 1;
				cols[3].k = zk;
				cols[3].c = tempIndices[0];
				cols[4].i = xi; // top
				cols[4].j = yj + 1;
				cols[4].k = zk;
				cols[4].c = tempIndices[0];
				cols[5].i = xi; // front
				cols[5].j = yj;
				cols[5].k = zk - 1;
				cols[5].c = tempIndices[0];
				cols[6].i = xi; // back
				cols[6].j = yj;
				cols[6].k = zk + 1;
				cols[6].c = tempIndices[0];

				ierr = MatSetValuesStencil(J, 1, &row, 7, cols, tempVals,
						ADD_VALUES);
				checkPetscError(ierr,
						"PetscSolver3DHandler::computeOffDiagonalJacobian: "
								"MatSetValuesStencil (temperature) failed.");

				// Get the partial derivatives for the diffusion
				diffusionHandler->computePartialsForDiffusion(network, diffVals,
						diffIndices, hxLeft, hxRight, xi - localXS, sy,
						yj - localYS, sz, zk - localZS);

				// Loop on the number of diffusion cluster to set the values in the Jacobian
				for (int i = 0; i < nDiff; i++) {
					// Set grid coordinate and component number for the row
					row.i = xi;
					row.j = yj;
					row.k = zk;
					row.c = diffIndices[i];

					// Set grid coordinates and component numbers for the columns
					// corresponding to the middle, left, right, bottom, top, front,
					// and back grid points
					cols[0].i = xi;					// middle
					cols[0].j = yj;
					cols[0].k = zk;
					cols[0].c = diffIndices[i];
					cols[1].i = xi - 1;					// left
					cols[1].j = yj;
					cols[1].k = zk;
					cols[1].c = diffIndices[i];
					cols[2].i = xi + 1;					// right
					cols[2].j = yj;
					cols[2].k = zk;
					cols[2].c = diffIndices[i];
					cols[3].i = xi;					// bottom
					cols[3].j = yj - 1;
					cols[3].k = zk;
					cols[3].c = diffIndices[i];
					cols[4].i = xi;					// top
					cols[4].j = yj + 1;
					cols[4].k = zk;
					cols[4].c = diffIndices[i];
					cols[5].i = xi;					// front
					cols[5].j = yj;
					cols[5].k = zk - 1;
					cols[5].c = diffIndices[i];
					cols[6].i = xi;					// back
					cols[6].j = yj;
					cols[6].k = zk + 1;
					cols[6].c = diffIndices[i];

					ierr = MatSetValuesStencil(J, 1, &row, 7, cols,
							diffVals + (7 * i), ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeOffDiagonalJacobian: "
									"MatSetValuesStencil (diffusion) failed.");
				}

				// Get the partial derivatives for the advection
				// Set the grid position
				gridPosition[0] = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
				for (int l = 0; l < advectionHandlers.size(); l++) {
					advectionHandlers[l]->computePartialsForAdvection(network,
							advecVals, advecIndices, gridPosition, hxLeft,
							hxRight, xi - localXS, hY, yj - localYS, hZ,
							zk - localZS);

					// Get the stencil indices to know where to put the partial derivatives in the Jacobian
					auto advecStencil =
							advectionHandlers[l]->getStencilForAdvection(
									gridPosition);

					// Get the number of advecting clusters
					nAdvec = advectionHandlers[l]->getNumberOfAdvecting();

					// Loop on the number of advecting cluster to set the values in the Jacobian
					for (int i = 0; i < nAdvec; i++) {
						// Set grid coordinate and component number for the row
						row.i = xi;
						row.j = yj;
						row.k = zk;
						row.c = advecIndices[i];

						// If we are on the sink, the partial derivatives are not the same
						// Both sides are giving their concentrations to the center
						if (advectionHandlers[l]->isPointOnSink(gridPosition)) {
							cols[0].i = xi - advecStencil[0]; // left?
							cols[0].j = yj - advecStencil[1]; // bottom?
							cols[0].k = zk - advecStencil[2]; // back?
							cols[0].c = advecIndices[i];
							cols[1].i = xi + advecStencil[0]; // right?
							cols[1].j = yj + advecStencil[1]; // top?
							cols[1].k = zk + advecStencil[2]; // front?
							cols[1].c = advecIndices[i];
						} else {
							// Set grid coordinates and component numbers for the columns
							// corresponding to the middle and other grid points
							cols[0].i = xi;							// middle
							cols[0].j = yj;
							cols[0].k = zk;
							cols[0].c = advecIndices[i];
							cols[1].i = xi + advecStencil[0];// left or right?
							cols[1].j = yj + advecStencil[1];// bottom or top?
							cols[1].k = zk + advecStencil[2];// back or front?
							cols[1].c = advecIndices[i];
						}

						// Update the matrix
						ierr = MatSetValuesStencil(J, 1, &row, 2, cols,
								advecVals + (2 * i), ADD_VALUES);
						checkPetscError(ierr,
								"PetscSolver3DHandler::computeOffDiagonalJacobian: "
										"MatSetValuesStencil (advection) failed.");
					}
				}
			}
		}
	}

	// Restore the array
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver3DHandler::computeOffDiagonalJacobian: "
			"DMDAVecRestoreArrayDOFRead (localC) failed.");

	return;
}

void PetscSolver3DHandler::computeDiagonalJacobian(TS &ts, Vec &localC, Mat &J,
		PetscReal ftime) {
	PetscErrorCode ierr;

	// Get the distributed array
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(ierr, "PetscSolver3DHandler::computeDiagonalJacobian: "
			"TSGetDM failed.");

	// Get pointers to vector data
	PetscScalar ****concs = nullptr;
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver3DHandler::computeDiagonalJacobian: "
			"DMDAVecGetArrayDOFRead failed.");

	// The degree of freedom is the size of the network
	const int dof = network.getDOF();

	// Pointer to the concentrations at a given grid point
	PetscScalar *concOffset = nullptr;

	// Arguments for MatSetValuesStencil called below
	MatStencil rowId;
	MatStencil colIds[dof];
	int pdColIdsVectorSize = 0;

	// Declarations for variables used in the loop
	double atomConc = 0.0, totalAtomConc = 0.0;
	xolotlCore::NDPoint<3> gridPosition { 0.0, 0.0, 0.0 };

	// Loop over the grid points
	for (int zk = frontOffset; zk < nZ - backOffset; zk++) {
		for (int yj = bottomOffset; yj < nY - topOffset; yj++) {

			// Computing the trapped atom concentration is only needed for the attenuation
			if (useAttenuation) {
				// Compute the total concentration of atoms contained in bubbles
				atomConc = 0.0;

				// Loop over grid points
				for (int xi = surfacePosition[yj][zk] + leftOffset;
						xi < nX - rightOffset; xi++) {
					// We are only interested in the helium near the surface
					if ((grid[xi] + grid[xi + 1]) / 2.0
							- grid[surfacePosition[yj][zk] + 1] > 2.0)
						continue;

					// Check if we are on the right processor
					if (xi >= localXS && xi < localXS + localXM && yj >= localYS
							&& yj < localYS + localYM && zk >= localZS
							&& zk < localZS + localZM) {
						// Get the concentrations at this grid point
						concOffset = concs[zk][yj][xi];
						// Copy data into the PSIClusterReactionNetwork
						network.updateConcentrationsFromArray(concOffset);

						// Sum the total atom concentration
						atomConc += network.getTotalTrappedAtomConcentration()
								* (grid[xi + 1] - grid[xi]);
					}
				}

				// Share the concentration with all the processes
				totalAtomConc = 0.0;
				auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
				MPI_Allreduce(&atomConc, &totalAtomConc, 1, MPI_DOUBLE, MPI_SUM,
						xolotlComm);

				// Set the disappearing rate in the modified TM handler
				mutationHandler->updateDisappearingRate(totalAtomConc);
			}

			// Skip if we are not on the right process
			if (yj < localYS || yj >= localYS + localYM || zk < localZS
					|| zk >= localZS + localZM)
				continue;

			// Set the grid position
			gridPosition[1] = yj * hY;
			gridPosition[2] = zk * hZ;

			for (int xi = localXS; xi < localXS + localXM; xi++) {
				// Boundary conditions
				// Everything to the left of the surface is empty
				if (xi < surfacePosition[yj][zk] + leftOffset
						|| xi > nX - 1 - rightOffset || yj < bottomOffset
						|| yj > nY - 1 - topOffset || zk < frontOffset
						|| zk > nZ - 1 - backOffset)
					continue;
				// Free surface GB
				bool skip = false;
				for (auto &pair : gbVector) {
					if (xi == std::get<0>(pair) && yj == std::get<1>(pair)
							&& zk == std::get<2>(pair)) {
						skip = true;
						break;
					}
				}
				if (skip)
					continue;

				// Set the grid fraction
				gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0
						- grid[surfacePosition[yj][zk] + 1])
						/ (grid[grid.size() - 1]
								- grid[surfacePosition[yj][zk] + 1]);

				// Get the temperature from the temperature handler
				concOffset = concs[zk][yj][xi];
				temperatureHandler->setTemperature(concOffset);
				double temperature = temperatureHandler->getTemperature(
						gridPosition, ftime);

				// Update the network if the temperature changed
				if (std::fabs(lastTemperature[xi + 1 - localXS] - temperature)
						> 0.1) {
					network.setTemperature(temperature, xi + 1 - localXS);
					// Update the modified trap-mutation rate that depends on the
					// network reaction rates
					mutationHandler->updateTrapMutationRate(network);
					lastTemperature[xi + 1 - localXS] = temperature;
				}

				// Copy data into the ReactionNetwork so that it can
				// compute the new concentrations.
				network.updateConcentrationsFromArray(concOffset);

				// ----- Take care of the reactions for all the reactants -----

				// Compute all the partial derivatives for the reactions
				network.computeAllPartials(reactionStartingIdx, reactionIndices,
						reactionVals, xi + 1 - localXS);

				// Update the column in the Jacobian that represents each DOF
				for (int i = 0; i < dof - 1; i++) {
					// Set grid coordinate and component number for the row
					rowId.i = xi;
					rowId.j = yj;
					rowId.k = zk;
					rowId.c = i;

					// Number of partial derivatives
					pdColIdsVectorSize = reactionSize[i];
					auto startingIdx = reactionStartingIdx[i];

					// Loop over the list of column ids
					for (int j = 0; j < pdColIdsVectorSize; j++) {
						// Set grid coordinate and component number for a column in the list
						colIds[j].i = xi;
						colIds[j].j = yj;
						colIds[j].k = zk;
						colIds[j].c = reactionIndices[startingIdx + j];
						// Get the partial derivative from the array of all of the partials
						reactingPartialsForCluster[j] = reactionVals[startingIdx
								+ j];
					}
					// Update the matrix
					ierr = MatSetValuesStencil(J, 1, &rowId, pdColIdsVectorSize,
							colIds, reactingPartialsForCluster.data(),
							ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeDiagonalJacobian: "
									"MatSetValuesStencil (reactions) failed.");
				}

				// ----- Take care of the modified trap-mutation for all the reactants -----

				// Store the total number of He clusters in the network for the
				// modified trap-mutation
				int nHelium = mutationHandler->getNumberOfMutating();

				// Arguments for MatSetValuesStencil called below
				MatStencil row, col;
				PetscScalar mutationVals[3 * nHelium];
				PetscInt mutationIndices[3 * nHelium];

				// Compute the partial derivative from modified trap-mutation at this grid point
				int nMutating = mutationHandler->computePartialsForTrapMutation(
						network, mutationVals, mutationIndices, xi - localXS,
						yj - localYS, zk - localZS);

				// Loop on the number of helium undergoing trap-mutation to set the values
				// in the Jacobian
				for (int i = 0; i < nMutating; i++) {
					// Set grid coordinate and component number for the row and column
					// corresponding to the helium cluster
					row.i = xi;
					row.j = yj;
					row.k = zk;
					row.c = mutationIndices[3 * i];
					col.i = xi;
					col.j = yj;
					col.k = zk;
					col.c = mutationIndices[3 * i];

					ierr = MatSetValuesStencil(J, 1, &row, 1, &col,
							mutationVals + (3 * i), ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeDiagonalJacobian: "
									"MatSetValuesStencil (He trap-mutation) failed.");

					// Set component number for the row
					// corresponding to the HeV cluster created through trap-mutation
					row.c = mutationIndices[(3 * i) + 1];

					ierr = MatSetValuesStencil(J, 1, &row, 1, &col,
							mutationVals + (3 * i) + 1, ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeDiagonalJacobian: "
									"MatSetValuesStencil (HeV trap-mutation) failed.");

					// Set component number for the row
					// corresponding to the interstitial created through trap-mutation
					row.c = mutationIndices[(3 * i) + 2];

					ierr = MatSetValuesStencil(J, 1, &row, 1, &col,
							mutationVals + (3 * i) + 2, ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeDiagonalJacobian: "
									"MatSetValuesStencil (I trap-mutation) failed.");
				}

				// ----- Take care of the re-solution for all the reactants -----

				// Store the total number of Xe clusters in the network
				int nXenon = resolutionHandler->getNumberOfReSoluting();

				// Arguments for MatSetValuesStencil called below
				PetscScalar resolutionVals[10 * nXenon];
				PetscInt resolutionIndices[5 * nXenon];
				MatStencil rowIds[5];

				// Compute the partial derivative from re-solution at this grid point
				int nResoluting =
						resolutionHandler->computePartialsForReSolution(network,
								resolutionVals, resolutionIndices, xi, localXS,
								yj, zk);

				// Loop on the number of xenon to set the values in the Jacobian
				for (int i = 0; i < nResoluting; i++) {
					// Set grid coordinate and component number for the row and column
					// corresponding to the clusters involved in re-solution
					rowIds[0].i = xi;
					rowIds[0].j = yj;
					rowIds[0].k = zk;
					rowIds[0].c = resolutionIndices[5 * i];
					rowIds[1].i = xi;
					rowIds[1].j = yj;
					rowIds[1].k = zk;
					rowIds[1].c = resolutionIndices[(5 * i) + 1];
					rowIds[2].i = xi;
					rowIds[2].j = yj;
					rowIds[2].k = zk;
					rowIds[2].c = resolutionIndices[(5 * i) + 2];
					rowIds[3].i = xi;
					rowIds[3].j = yj;
					rowIds[3].k = zk;
					rowIds[3].c = resolutionIndices[(5 * i) + 3];
					rowIds[4].i = xi;
					rowIds[4].j = yj;
					rowIds[4].k = zk;
					rowIds[4].c = resolutionIndices[(5 * i) + 4];
					colIds[0].i = xi;
					colIds[0].j = yj;
					colIds[0].k = zk;
					colIds[0].c = resolutionIndices[5 * i];
					colIds[1].i = xi;
					colIds[1].j = yj;
					colIds[1].k = zk;
					colIds[1].c = resolutionIndices[(5 * i) + 1];
					ierr = MatSetValuesStencil(J, 5, rowIds, 2, colIds,
							resolutionVals + (10 * i), ADD_VALUES);
					checkPetscError(ierr,
							"PetscSolver3DHandler::computeDiagonalJacobian: "
									"MatSetValuesStencil (Xe re-solution) failed.");
				}
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver3DHandler::computeDiagonalJacobian: "
			"DMDAVecRestoreArrayDOFRead failed.");

	return;
}

} /* end namespace xolotlSolver */
