// Includes
#include <PetscSolver1DHandler.h>
#include <MathUtils.h>
#include <Constants.h>
#include <experimental/PSIReactionNetwork.h>

namespace xcore = xolotlCore;

namespace xolotlSolver {

void PetscSolver1DHandler::createSolverContext(DM &da) {
	PetscErrorCode ierr;

	// Degrees of freedom is the total number of clusters in the network
	const int dof = expNetwork.getDOF();

	// Set the position of the surface
	surfacePosition = 0;
	if (movingSurface)
		surfacePosition = (int) (nX * portion / 100.0);

	// Generate the grid in the x direction
	generateGrid(nX, hX, surfacePosition);

	// Now that the grid was generated, we can update the surface position
	// if we are using a restart file
	if (not networkName.empty() and movingSurface) {

		xolotlCore::XFile xfile(networkName);
		auto concGroup =
				xfile.getGroup<xolotlCore::XFile::ConcentrationGroup>();
		if (concGroup and concGroup->hasTimesteps()) {
			auto tsGroup = concGroup->getLastTimestepGroup();
			assert(tsGroup);
			surfacePosition = tsGroup->readSurface1D();
		}
	}

	// Prints the grid on one process
	int procId;
	MPI_Comm_rank(PETSC_COMM_WORLD, &procId);
	if (procId == 0) {
		for (int i = 1; i < grid.size() - 1; i++) {
			std::cout << grid[i] - grid[surfacePosition + 1] << " ";
		}
		std::cout << std::endl;
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Create distributed array (DMDA) to manage parallel grid and vectors
	 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, nX, dof + 1, 1,
	NULL, &da);
	checkPetscError(ierr, "PetscSolver1DHandler::createSolverContext: "
			"DMDACreate1d failed.");
	ierr = DMSetFromOptions(da);
	checkPetscError(ierr,
			"PetscSolver1DHandler::createSolverContext: DMSetFromOptions failed.");
	ierr = DMSetUp(da);
	checkPetscError(ierr,
			"PetscSolver1DHandler::createSolverContext: DMSetUp failed.");

	// Initialize the surface of the first advection handler corresponding to the
	// advection toward the surface (or a dummy one if it is deactivated)
	advectionHandlers[0]->setLocation(grid[surfacePosition + 1] - grid[1]);

	/*  The only spatial coupling in the Jacobian is due to diffusion.
	 *  The ofill (thought of as a dof by dof 2d (row-oriented) array represents
	 *  the nonzero coupling between degrees of freedom at one point with degrees
	 *  of freedom on the adjacent point to the left or right. A 1 at i,j in the
	 *  ofill array indicates that the degree of freedom i at a point is coupled
	 *  to degree of freedom j at the adjacent point.
	 *  In this case ofill has only a few diagonal entries since the only spatial
	 *  coupling is regular diffusion.
	 */
	xolotlCore::experimental::IReactionNetwork::SparseFillMap ofill;

	// Initialize the temperature handler
	temperatureHandler->initializeTemperature(dof, ofill, dfill);

	// Fill ofill, the matrix of "off-diagonal" elements that represents diffusion
	diffusionHandler->initializeOFill(expNetwork, ofill);
	// Loop on the advection handlers to account the other "off-diagonal" elements
	for (int i = 0; i < advectionHandlers.size(); i++) {
		advectionHandlers[i]->initialize(expNetwork, ofill);
	}

	// Get the local boundaries
	PetscInt xs, xm;
	ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL);
	checkPetscError(ierr, "PetscSolver1DHandler::initializeConcentration: "
			"DMDAGetCorners failed.");

	// Initialize the modified trap-mutation handler here
	// because it adds connectivity
	mutationHandler->initialize(expNetwork, dfill, xm);
	mutationHandler->initializeIndex1D(surfacePosition, expNetwork,
			advectionHandlers, grid, xm, xs);

	// Tell the network the number of grid points on this process with ghosts
	// TODO: do we need the ghost points?
	expNetwork.setGridSize(xm + 2);

	// Get the diagonal fill
	auto nPartials = expNetwork.getDiagonalFill(dfill);

	// Load up the block fills
	auto dfillsparse = ConvertToPetscSparseFillMap(dof + 1, dfill);
	auto ofillsparse = ConvertToPetscSparseFillMap(dof + 1, ofill);
	ierr = DMDASetBlockFillsSparse(da, dfillsparse.data(), ofillsparse.data());
	checkPetscError(ierr, "PetscSolver1DHandler::createSolverContext: "
			"DMDASetBlockFills failed.");

	// Initialize the arrays for the reaction partial derivatives
	expVals = Kokkos::View<double*>("solverPartials", nPartials);

	// Set the size of the partial derivatives vectors
	reactingPartialsForCluster.resize(dof, 0.0);

	return;
}

void PetscSolver1DHandler::initializeConcentration(DM &da, Vec &C) {
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar **concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver1DHandler::initializeConcentration: "
			"DMDAVecGetArrayDOF failed.");

	// Get the local boundaries
	PetscInt xs, xm;
	ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL);
	checkPetscError(ierr, "PetscSolver1DHandler::initializeConcentration: "
			"DMDAGetCorners failed.");

	// Initialize the last temperature at each grid point on this process
	for (int i = 0; i < xm + 2; i++) {
		temperature.push_back(0.0);
	}

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
	temperatureHandler->updateSurfacePosition(surfacePosition);

	// Initialize the flux handler
	fluxHandler->initializeFluxHandler(expNetwork, surfacePosition, grid);

	// Initialize the grid for the diffusion
	diffusionHandler->initializeDiffusionGrid(advectionHandlers, grid, xm, xs);

	// Initialize the grid for the advection
	advectionHandlers[0]->initializeAdvectionGrid(advectionHandlers, grid, xm,
			xs);

	// Pointer for the concentration vector at a specific grid point
	PetscScalar *concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	// + the super clusters
	const int dof = expNetwork.getDOF();

	// Get the single vacancy ID
	auto singleVacancyCluster = expNetwork.getSingleVacancy();
	auto vacancyIndex = NetworkType::invalidIndex();
	if (singleVacancyCluster.getId() != NetworkType::invalidIndex())
		vacancyIndex = singleVacancyCluster.getId();

	// Loop on all the grid points
	for (PetscInt i = xs - 1; i <= xs + xm; i++) {
		// Temperature
		xolotlCore::Point<3> gridPosition { 0.0, 0.0, 0.0 };
		if (i < 0)
			gridPosition[0] = (grid[0] - grid[surfacePosition + 1])
					/ (grid[grid.size() - 1] - grid[surfacePosition + 1]);
		else
			gridPosition[0] = ((grid[i] + grid[i + 1]) / 2.0
					- grid[surfacePosition + 1])
					/ (grid[grid.size() - 1] - grid[surfacePosition + 1]);
		auto temp = temperatureHandler->getTemperature(gridPosition, 0.0);
		temperature[i - xs + 1] = temp;

		// Boundary conditions
		if (i < xs || i >= xs + xm)
			continue;

		concOffset = concentrations[i];
		concOffset[dof] = temp;

		// Loop on all the clusters to initialize at 0.0
		for (int n = 0; n < dof; n++) {
			concOffset[n] = 0.0;
		}

		// Initialize the vacancy concentration
		if (i >= surfacePosition + leftOffset
				and vacancyIndex != NetworkType::invalidIndex()
				and not hasConcentrations and i < nX - rightOffset) {
			concOffset[vacancyIndex] = initialVConc;
		}
	}

	// If the concentration must be set from the HDF5 file
	if (hasConcentrations) {

		// Read the concentrations from the HDF5 file for
		// each of our grid points.
		assert(concGroup);
		auto tsGroup = concGroup->getLastTimestepGroup();
		assert(tsGroup);
		auto myConcs = tsGroup->readConcentrations(*xfile, xs, xm);

		// Apply the concentrations we just read.
		for (auto i = 0; i < xm; ++i) {
			concOffset = concentrations[xs + i];

			for (auto const &currConcData : myConcs[i]) {
				concOffset[currConcData.first] = currConcData.second;
			}
			// Get the temperature
			double temp = myConcs[i][myConcs[i].size() - 1].second;
			temperature[i + 1] = temp;
		}
	}

	// Update the network with the temperature
	expNetwork.setTemperatures(temperature);
	expNetwork.syncClusterDataOnHost();
	// Update the modified trap-mutation rate
	// that depends on the network reaction rates
	mutationHandler->updateTrapMutationRate(expNetwork.getLargestRate());

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr, "PetscSolver1DHandler::initializeConcentration: "
			"DMDAVecRestoreArrayDOF failed.");

	return;
}

void PetscSolver1DHandler::updateConcentration(TS &ts, Vec &localC, Vec &F,
		PetscReal ftime) {
	PetscErrorCode ierr;

	// Get the local data vector from PETSc
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(ierr, "PetscSolver1DHandler::updateConcentration: "
			"TSGetDM failed.");

	// Pointers to the PETSc arrays that start at the beginning (xs) of the
	// local array!
	PetscScalar **concs = nullptr, **updatedConcs = nullptr;
	// Get pointers to vector data
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver1DHandler::updateConcentration: "
			"DMDAVecGetArrayDOFRead (localC) failed.");
	ierr = DMDAVecGetArrayDOF(da, F, &updatedConcs);
	checkPetscError(ierr, "PetscSolver1DHandler::updateConcentration: "
			"DMDAVecGetArrayDOF (F) failed.");

	// Get local grid boundaries
	PetscInt xs, xm;
	ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL);
	checkPetscError(ierr, "PetscSolver1DHandler::updateConcentration: "
			"DMDAGetCorners failed.");

	// The following pointers are set to the first position in the conc or
	// updatedConc arrays that correspond to the beginning of the data for the
	// current grid point. They are accessed just like regular arrays.
	PetscScalar *concOffset = nullptr, *updatedConcOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	const int dof = expNetwork.getDOF();

	// Computing the trapped atom concentration is only needed for the attenuation
	if (useAttenuation) {
		// Compute the total concentration of atoms contained in bubbles
		double atomConc = 0.0;

		// Loop over grid points to get the atom concentration
		// near the surface
		for (int xi = xs; xi < xs + xm; xi++) {
			// Boundary conditions
			if (xi < surfacePosition + leftOffset || xi > nX - 1 - rightOffset)
				continue;

			// We are only interested in the helium near the surface
			if ((grid[xi] + grid[xi + 1]) / 2.0 - grid[surfacePosition + 1]
					> 2.0)
				continue;

			// Get the concentrations at this grid point
			concOffset = concs[xi];

			// Sum the total atom concentration
			using NetworkType =
			xolotlCore::experimental::PSIReactionNetwork<xolotlCore::experimental::PSIFullSpeciesList>;
			using Spec = typename NetworkType::Species;
			using HostUnmanaged =
			Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(concOffset, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);
			// TODO: how to not have to cast the network here?
			auto &psiNetwork = dynamic_cast<NetworkType&>(expNetwork);
			atomConc += psiNetwork.getTotalTrappedAtomConcentration(dConcs,
					Spec::He, 0) * (grid[xi + 1] - grid[xi]);
		}

		// Share the concentration with all the processes
		double totalAtomConc = 0.0;
		MPI_Allreduce(&atomConc, &totalAtomConc, 1, MPI_DOUBLE, MPI_SUM,
				MPI_COMM_WORLD);

		// Set the disappearing rate in the modified TM handler
		mutationHandler->updateDisappearingRate(totalAtomConc);
	}

	// Declarations for variables used in the loop
	double **concVector = new double*[3];
	xolotlCore::Point<3> gridPosition { 0.0, 0.0, 0.0 };

	// Loop over grid points first for the temperature, including the ghost points
	bool tempHasChanged = false;
	for (PetscInt xi = xs - 1; xi <= xs + xm; xi++) {

		// Heat condition
		if (xi == surfacePosition && xi >= xs && xi < xs + xm) {
			// Compute the old and new array offsets
			concOffset = concs[xi];
			updatedConcOffset = updatedConcs[xi];

			// Fill the concVector with the pointer to the middle, left, and right grid points
			concVector[0] = concOffset; // middle
			concVector[1] = concs[xi - 1]; // left
			concVector[2] = concs[xi + 1]; // right

			// Compute the left and right hx
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi - 1 >= 0 && xi < nX) {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			} else if (xi == -1) {
				hxLeft = 0.0;
				hxRight = (grid[xi + 2] + grid[xi + 1]) / 2.0;
			} else if (xi - 1 < 0) {
				hxLeft = (grid[xi + 1] + grid[xi]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			} else {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 1] - grid[xi]) / 2;
			}

			temperatureHandler->computeTemperature(concVector,
					updatedConcOffset, hxLeft, hxRight, xi);
		}

		// Boundary conditions
		// Everything to the left of the surface is empty
		if (xi < surfacePosition + leftOffset || xi > nX - 1 - rightOffset) {
			continue;
		}
		// Free surface GB
		bool skip = false;
		for (auto &pair : gbVector) {
			if (xi == std::get<0>(pair)) {
				skip = true;
				break;
			}
		}
		if (skip)
			continue;

		// Compute the old and new array offsets
		concOffset = concs[xi];
		updatedConcOffset = updatedConcs[xi];

		// Fill the concVector with the pointer to the middle, left, and right grid points
		concVector[0] = concOffset; // middle
		concVector[1] = concs[xi - 1]; // left
		concVector[2] = concs[xi + 1]; // right

		// Compute the left and right hx
		double hxLeft = 0.0, hxRight = 0.0;
		if (xi - 1 >= 0 && xi < nX) {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else if (xi - 1 < 0) {
			hxLeft = (grid[xi + 1] + grid[xi]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 1] - grid[xi]) / 2;
		}

		// Set the grid fraction
		gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0
				- grid[surfacePosition + 1])
				/ (grid[grid.size() - 1] - grid[surfacePosition + 1]);

		// Get the temperature from the temperature handler
		temperatureHandler->setTemperature(concOffset);
		double temp = temperatureHandler->getTemperature(gridPosition, ftime);

		// Update the network if the temperature changed
		if (std::fabs(temperature[xi + 1 - xs] - temp) > 0.1) {
			temperature[xi + 1 - xs] = temp;
			tempHasChanged = true;
		}

		// ---- Compute the temperature over the locally owned part of the grid -----
		if (xi >= xs && xi < xs + xm) {
			temperatureHandler->computeTemperature(concVector,
					updatedConcOffset, hxLeft, hxRight, xi);
		}
	}

	if (tempHasChanged) {
		// Update the network with the temperature
		expNetwork.setTemperatures(temperature);
		expNetwork.syncClusterDataOnHost();
		// Update the modified trap-mutation rate
		// that depends on the network reaction rates
		// TODO: is this just the local largest rate? Is it correct?
		mutationHandler->updateTrapMutationRate(expNetwork.getLargestRate());
	}

	// Loop over grid points computing ODE terms for each grid point
	for (PetscInt xi = xs; xi < xs + xm; xi++) {
		// Compute the old and new array offsets
		concOffset = concs[xi];
		updatedConcOffset = updatedConcs[xi];

		// Fill the concVector with the pointer to the middle, left, and right grid points
		concVector[0] = concOffset; // middle
		concVector[1] = concs[xi - 1]; // left
		concVector[2] = concs[xi + 1]; // right

		// Compute the left and right hx
		double hxLeft = 0.0, hxRight = 0.0;
		if (xi - 1 >= 0 && xi < nX) {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else if (xi - 1 < 0) {
			hxLeft = (grid[xi + 1] + grid[xi]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 1] - grid[xi]) / 2;
		}

		// Boundary conditions
		// Everything to the left of the surface is empty
		if (xi < surfacePosition + leftOffset || xi > nX - 1 - rightOffset) {
			continue;
		}
		// Free surface GB
		bool skip = false;
		for (auto &pair : gbVector) {
			if (xi == std::get<0>(pair)) {
				skip = true;
				break;
			}
		}
		if (skip)
			continue;

		// ----- Account for flux of incoming particles -----
		fluxHandler->computeIncidentFlux(ftime, updatedConcOffset, xi,
				surfacePosition);

		// ---- Compute diffusion over the locally owned part of the grid -----
		diffusionHandler->computeDiffusion(expNetwork, concVector,
				updatedConcOffset, hxLeft, hxRight, xi - xs);

		// ---- Compute advection over the locally owned part of the grid -----
		// Set the grid position
		gridPosition[0] = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
		for (int i = 0; i < advectionHandlers.size(); i++) {
			advectionHandlers[i]->computeAdvection(expNetwork, gridPosition,
					concVector, updatedConcOffset, hxLeft, hxRight, xi - xs);
		}

		// ----- Compute the modified trap-mutation over the locally owned part of the grid -----
		mutationHandler->computeTrapMutation(expNetwork, concOffset,
				updatedConcOffset, xi - xs);

		// ----- Compute the reaction fluxes over the locally owned part of the grid -----
		using HostUnmanaged =
		Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
		auto hConcs = HostUnmanaged(concOffset, dof + 1);
		auto dConcs = Kokkos::View<double*>("Concentrations", dof + 1);
		deep_copy(dConcs, hConcs);
		auto hFlux = HostUnmanaged(updatedConcOffset, dof + 1);
		auto dFlux = Kokkos::View<double*>("Fluxes", dof + 1);
		deep_copy(dFlux, hFlux);
		fluxCounter->increment();
		fluxTimer->start();
		expNetwork.computeAllFluxes(dConcs, dFlux, xi + 1 - xs);
		fluxTimer->stop();
		deep_copy(hFlux, dFlux);
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver1DHandler::updateConcentration: "
			"DMDAVecRestoreArrayDOFRead (localC) failed.");
	ierr = DMDAVecRestoreArrayDOF(da, F, &updatedConcs);
	checkPetscError(ierr, "PetscSolver1DHandler::updateConcentration: "
			"DMDAVecRestoreArrayDOF (F) failed.");

	// Clear memory
	delete[] concVector;

	return;
}

void PetscSolver1DHandler::computeJacobian(TS &ts, Vec &localC, Mat &J,
		PetscReal ftime) {
	PetscErrorCode ierr;

	// Get the distributed array
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
			"TSGetDM failed.");

	// Get pointers to vector data
	PetscScalar **concs = nullptr;
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
			"DMDAVecGetArrayDOFRead failed.");

	// Get local grid boundaries
	PetscInt xs, xm;
	ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL);
	checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
			"DMDAGetCorners failed.");

	// Pointer to the concentrations at a given grid point
	PetscScalar *concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	const int dof = expNetwork.getDOF();

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
	MatStencil row, cols[3];
	PetscScalar tempVals[3];
	PetscInt tempIndices[1];
	PetscScalar diffVals[3 * nDiff];
	PetscInt diffIndices[nDiff];
	PetscScalar advecVals[2 * nAdvec];
	PetscInt advecIndices[nAdvec];
	xolotlCore::Point<3> gridPosition { 0.0, 0.0, 0.0 };

	/*
	 Loop over grid points for the temperature, including ghosts
	 */
	bool tempHasChanged = false;
	for (PetscInt xi = xs - 1; xi <= xs + xm; xi++) {
		// Compute the left and right hx
		double hxLeft = 0.0, hxRight = 0.0;
		if (xi - 1 >= 0 && xi < nX) {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else if (xi == -1) {
			hxLeft = 0.0;
			hxRight = (grid[xi + 2] + grid[xi + 1]) / 2.0;
		} else if (xi - 1 < 0) {
			hxLeft = (grid[xi + 1] + grid[xi]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 1] - grid[xi]) / 2;
		}

		// Heat condition
		if (xi == surfacePosition && xi >= xs && xi < xs + xm) {
			// Get the partial derivatives for the temperature
			auto setValues = temperatureHandler->computePartialsForTemperature(
					tempVals, tempIndices, hxLeft, hxRight, xi);

			if (setValues) {
				// Set grid coordinate and component number for the row
				row.i = xi;
				row.c = tempIndices[0];

				// Set grid coordinates and component numbers for the columns
				// corresponding to the middle, left, and right grid points
				cols[0].i = xi; // middle
				cols[0].c = tempIndices[0];
				cols[1].i = xi - 1; // left
				cols[1].c = tempIndices[0];
				cols[2].i = xi + 1; // right
				cols[2].c = tempIndices[0];

				ierr = MatSetValuesStencil(J, 1, &row, 3, cols, tempVals,
						ADD_VALUES);
				checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
						"MatSetValuesStencil (temperature) failed.");
			}
		}

		// Boundary conditions
		// Everything to the left of the surface is empty
		if (xi < surfacePosition + leftOffset || xi > nX - 1 - rightOffset)
			continue;
		// Free surface GB
		bool skip = false;
		for (auto &pair : gbVector) {
			if (xi == std::get<0>(pair)) {
				skip = true;
				break;
			}
		}
		if (skip)
			continue;

		// Get the concentrations at this grid point
		concOffset = concs[xi];

		// Set the grid fraction
		gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0
				- grid[surfacePosition + 1])
				/ (grid[grid.size() - 1] - grid[surfacePosition + 1]);

		// Get the temperature from the temperature handler
		temperatureHandler->setTemperature(concOffset);
		double temp = temperatureHandler->getTemperature(gridPosition, ftime);

		// Update the network if the temperature changed
		if (std::fabs(temperature[xi + 1 - xs] - temp) > 0.1) {
			temperature[xi + 1 - xs] = temp;
			tempHasChanged = true;
		}

		// Get the partial derivatives for the temperature
		if (xi >= xs && xi < xs + xm) {
			auto setValues = temperatureHandler->computePartialsForTemperature(
					tempVals, tempIndices, hxLeft, hxRight, xi);

			if (setValues) {
				// Set grid coordinate and component number for the row
				row.i = xi;
				row.c = tempIndices[0];

				// Set grid coordinates and component numbers for the columns
				// corresponding to the middle, left, and right grid points
				cols[0].i = xi; // middle
				cols[0].c = tempIndices[0];
				cols[1].i = xi - 1; // left
				cols[1].c = tempIndices[0];
				cols[2].i = xi + 1; // right
				cols[2].c = tempIndices[0];

				ierr = MatSetValuesStencil(J, 1, &row, 3, cols, tempVals,
						ADD_VALUES);
				checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
						"MatSetValuesStencil (temperature) failed.");
			}
		}
	}

	if (tempHasChanged) {
		// Update the network with the temperature
		expNetwork.setTemperatures(temperature);
		expNetwork.syncClusterDataOnHost();
		// Update the modified trap-mutation rate
		// that depends on the network reaction rates
		// TODO: is this just the local largest rate? Is it correct?
		mutationHandler->updateTrapMutationRate(expNetwork.getLargestRate());
	}

	// Computing the trapped atom concentration is only needed for the attenuation
	if (useAttenuation) {
		// Compute the total concentration of atoms contained in bubbles
		double atomConc = 0.0;

		// Loop over grid points to get the atom concentration
		// near the surface
		for (int xi = xs; xi < xs + xm; xi++) {
			// Boundary conditions
			if (xi < surfacePosition + leftOffset || xi > nX - 1 - rightOffset)
				continue;

			// We are only interested in the helium near the surface
			if ((grid[xi] + grid[xi + 1]) / 2.0 - grid[surfacePosition + 1]
					> 2.0)
				continue;

			// Get the concentrations at this grid point
			concOffset = concs[xi];

			// Sum the total atom concentration
			using NetworkType =
			xolotlCore::experimental::PSIReactionNetwork<xolotlCore::experimental::PSIFullSpeciesList>;
			using Spec = typename NetworkType::Species;
			using HostUnmanaged =
			Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(concOffset, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);
			// TODO: how to not have to cast the network here?
			auto &psiNetwork = dynamic_cast<NetworkType&>(expNetwork);
			atomConc += psiNetwork.getTotalTrappedAtomConcentration(dConcs,
					Spec::He, 0) * (grid[xi + 1] - grid[xi]);
		}

		// Share the concentration with all the processes
		double totalAtomConc = 0.0;
		MPI_Allreduce(&atomConc, &totalAtomConc, 1, MPI_DOUBLE, MPI_SUM,
				MPI_COMM_WORLD);

		// Set the disappearing rate in the modified TM handler
		mutationHandler->updateDisappearingRate(totalAtomConc);
	}

	// Arguments for MatSetValuesStencil called below
	MatStencil rowId;
	MatStencil colIds[dof];
	int pdColIdsVectorSize = 0;

	// Loop over the grid points
	for (PetscInt xi = xs; xi < xs + xm; xi++) {
		// Boundary conditions
		// Everything to the left of the surface is empty
		if (xi < surfacePosition + leftOffset || xi > nX - 1 - rightOffset)
			continue;

		// Free surface GB
		bool skip = false;
		for (auto &pair : gbVector) {
			if (xi == std::get<0>(pair)) {
				skip = true;
				break;
			}
		}
		if (skip)
			continue;
		// Compute the left and right hx
		double hxLeft = 0.0, hxRight = 0.0;
		if (xi - 1 >= 0 && xi < nX) {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else if (xi - 1 < 0) {
			hxLeft = (grid[xi + 1] + grid[xi]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 1] - grid[xi]) / 2;
		}

		// Get the partial derivatives for the diffusion
		diffusionHandler->computePartialsForDiffusion(expNetwork, diffVals,
				diffIndices, hxLeft, hxRight, xi - xs);

		// Loop on the number of diffusion cluster to set the values in the Jacobian
		for (int i = 0; i < nDiff; i++) {
			// Set grid coordinate and component number for the row
			row.i = xi;
			row.c = diffIndices[i];

			// Set grid coordinates and component numbers for the columns
			// corresponding to the middle, left, and right grid points
			cols[0].i = xi; // middle
			cols[0].c = diffIndices[i];
			cols[1].i = xi - 1; // left
			cols[1].c = diffIndices[i];
			cols[2].i = xi + 1; // right
			cols[2].c = diffIndices[i];

			ierr = MatSetValuesStencil(J, 1, &row, 3, cols, diffVals + (3 * i),
					ADD_VALUES);
			checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
					"MatSetValuesStencil (diffusion) failed.");
		}

		// Get the partial derivatives for the advection
		// Set the grid position
		gridPosition[0] = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
		for (int l = 0; l < advectionHandlers.size(); l++) {
			advectionHandlers[l]->computePartialsForAdvection(expNetwork,
					advecVals, advecIndices, gridPosition, hxLeft, hxRight,
					xi - xs);

			// Get the stencil indices to know where to put the partial derivatives in the Jacobian
			auto advecStencil = advectionHandlers[l]->getStencilForAdvection(
					gridPosition);

			// Get the number of advecting clusters
			nAdvec = advectionHandlers[l]->getNumberOfAdvecting();

			// Loop on the number of advecting cluster to set the values in the Jacobian
			for (int i = 0; i < nAdvec; i++) {
				// Set grid coordinate and component number for the row
				row.i = xi;
				row.c = advecIndices[i];

				// If we are on the sink, the partial derivatives are not the same
				// Both sides are giving their concentrations to the center
				if (advectionHandlers[l]->isPointOnSink(gridPosition)) {
					cols[0].i = xi - advecStencil[0]; // left?
					cols[0].c = advecIndices[i];
					cols[1].i = xi + advecStencil[0]; // right?
					cols[1].c = advecIndices[i];
				} else {
					// Set grid coordinates and component numbers for the columns
					// corresponding to the middle and other grid points
					cols[0].i = xi; // middle
					cols[0].c = advecIndices[i];
					cols[1].i = xi + advecStencil[0]; // left or right
					cols[1].c = advecIndices[i];
				}

				// Update the matrix
				ierr = MatSetValuesStencil(J, 1, &row, 2, cols,
						advecVals + (2 * i), ADD_VALUES);
				checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
						"MatSetValuesStencil (advection) failed.");
			}
		}

		// Get the concentrations at this grid point
		concOffset = concs[xi];

		// Compute all the partial derivatives for the reactions
		using HostUnmanaged =
		Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
		auto hConcs = HostUnmanaged(concOffset, dof + 1);
		auto dConcs = Kokkos::View<double*>("Concentrations", dof + 1);
		deep_copy(dConcs, hConcs);
		partialDerivativeCounter->increment();
		partialDerivativeTimer->start();
		expNetwork.computeAllPartials(dConcs, expVals, xi + 1 - xs);
		partialDerivativeTimer->stop();
		auto hPartials = create_mirror_view(expVals);
		deep_copy(hPartials, expVals);

		// Variable for the loop on reactants
		int startingIdx = 0;

		// Update the column in the Jacobian that represents each DOF
		for (int i = 0; i < dof; i++) {
			// Set grid coordinate and component number for the row
			rowId.i = xi;
			rowId.c = i;

			// Number of partial derivatives
			auto rowIter = dfill.find(i);
			if (rowIter != dfill.end()) {
				const auto &row = rowIter->second;
				pdColIdsVectorSize = row.size();

				// Loop over the list of column ids
				for (int j = 0; j < pdColIdsVectorSize; j++) {
					// Set grid coordinate and component number for a column in the list
					colIds[j].i = xi;
					colIds[j].c = row[j];
					// Get the partial derivative from the array of all of the partials
					reactingPartialsForCluster[j] = hPartials(startingIdx + j);
				}
				// Update the matrix
				ierr = MatSetValuesStencil(J, 1, &rowId, pdColIdsVectorSize,
						colIds, reactingPartialsForCluster.data(), ADD_VALUES);
				checkPetscError(ierr, "PetscSolverExpHandler::computeJacobian: "
						"MatSetValuesStencil (reactions) failed.");

				// Increase the starting index
				startingIdx += pdColIdsVectorSize;
			}
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
				expNetwork, concOffset, mutationVals, mutationIndices, xi - xs);

		// Loop on the number of helium undergoing trap-mutation to set the values
		// in the Jacobian
		for (int i = 0; i < nMutating; i++) {
			// Set grid coordinate and component number for the row and column
			// corresponding to the helium cluster
			row.i = xi;
			row.c = mutationIndices[3 * i];
			col.i = xi;
			col.c = mutationIndices[3 * i];

			ierr = MatSetValuesStencil(J, 1, &row, 1, &col,
					mutationVals + (3 * i), ADD_VALUES);
			checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
					"MatSetValuesStencil (He trap-mutation) failed.");

			// Set component number for the row
			// corresponding to the HeV cluster created through trap-mutation
			row.c = mutationIndices[(3 * i) + 1];

			ierr = MatSetValuesStencil(J, 1, &row, 1, &col,
					mutationVals + (3 * i) + 1, ADD_VALUES);
			checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
					"MatSetValuesStencil (HeV trap-mutation) failed.");

			// Set component number for the row
			// corresponding to the interstitial created through trap-mutation
			row.c = mutationIndices[(3 * i) + 2];

			ierr = MatSetValuesStencil(J, 1, &row, 1, &col,
					mutationVals + (3 * i) + 2, ADD_VALUES);
			checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
					"MatSetValuesStencil (I trap-mutation) failed.");
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr, "PetscSolver1DHandler::computeJacobian: "
			"DMDAVecRestoreArrayDOFRead failed.");

	return;
}

} /* end namespace xolotlSolver */
