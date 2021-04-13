// Includes
#include <xolotl/core/Constants.h>
#include <xolotl/core/network/IPSIReactionNetwork.h>
#include <xolotl/core/network/NEReactionNetwork.h>
#include <xolotl/solver/handler/PetscSolver2DHandler.h>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace solver
{
namespace handler
{
void
PetscSolver2DHandler::createSolverContext(DM& da)
{
	PetscErrorCode ierr;

	// Degrees of freedom is the total number of clusters in the network
	// + moments
	const auto dof = network.getDOF();

	// Set the position of the surface
	for (auto j = 0; j < nY; j++) {
		surfacePosition.push_back(0);
		if (movingSurface)
			surfacePosition[j] = (IdType)(nX * portion / 100.0);
	}

	// We can update the surface position
	// if we are using a restart file
	if (not networkName.empty() and movingSurface) {
		io::XFile xfile(networkName);
		auto concGroup = xfile.getGroup<io::XFile::ConcentrationGroup>();
		if (concGroup and concGroup->hasTimesteps()) {
			auto tsGroup = concGroup->getLastTimestepGroup();
			assert(tsGroup);
			auto surfaceIndices = tsGroup->readSurface2D();

			// Set the actual surface positions
			for (auto i = 0; i < surfaceIndices.size(); i++) {
				surfacePosition[i] = surfaceIndices[i];
			}
		}
	}

	// Prints info on one process
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);
	if (procId == 0) {
		std::cout << "SolverHandler: 2D simulation with surface BC: ";
		std::string bcString = "periodic";
		if (isMirror)
			bcString = "mirror";
		if (leftOffset == 1)
			std::cout << "free surface";
		else
			std::cout << bcString;
		std::cout << ", bulk BC: ";
		if (rightOffset == 1)
			std::cout << "free surface";
		else
			std::cout << bcString;
		std::cout << ", left BC: ";
		if (topOffset == 1)
			std::cout << "free surface";
		else
			std::cout << "periodic";
		std::cout << ", right BC: ";
		if (bottomOffset == 1)
			std::cout << "free surface";
		else
			std::cout << "periodic";
		std::cout << ", grid (nm): ";
		for (auto i = 1; i < grid.size() - 1; i++) {
			std::cout << grid[i] - grid[surfacePosition[0] + 1] << " ";
		}
		std::cout << std::endl;
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Create distributed array (DMDA) to manage parallel grid and vectors
	 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	if (isMirror) {
		ierr = DMDACreate2d(xolotlComm, DM_BOUNDARY_MIRROR,
			DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, nX, nY, PETSC_DECIDE,
			PETSC_DECIDE, dof + 1, 1, NULL, NULL, &da);
		checkPetscError(ierr,
			"PetscSolver2DHandler::createSolverContext: "
			"DMDACreate2d failed.");
	}
	else {
		ierr = DMDACreate2d(xolotlComm, DM_BOUNDARY_PERIODIC,
			DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, nX, nY, PETSC_DECIDE,
			PETSC_DECIDE, dof + 1, 1, NULL, NULL, &da);
		checkPetscError(ierr,
			"PetscSolver2DHandler::createSolverContext: "
			"DMDACreate2d failed.");
	}
	ierr = DMSetFromOptions(da);
	checkPetscError(ierr,
		"PetscSolver2DHandler::createSolverContext: DMSetFromOptions failed.");
	ierr = DMSetUp(da);
	checkPetscError(
		ierr, "PetscSolver2DHandler::createSolverContext: DMSetUp failed.");

	// Initialize the surface of the first advection handler corresponding to
	// the advection toward the surface
	advectionHandlers[0]->setLocation(grid[surfacePosition[0] + 1] - grid[1]);

	/* The ofill (thought of as a dof by dof 2d (row-oriented) array represents
	 * the nonzero coupling between degrees of freedom at one point with
	 * degrees of freedom on the adjacent point.
	 */
	core::network::IReactionNetwork::SparseFillMap ofill;

	// Initialize the temperature handler
	temperatureHandler->initializeTemperature(dof, ofill, dfill);

	// Fill ofill, the matrix of "off-diagonal" elements that represents
	// diffusion
	diffusionHandler->initializeOFill(network, ofill);
	// Loop on the advection handlers to account the other "off-diagonal"
	// elements
	for (auto i = 0; i < advectionHandlers.size(); i++) {
		advectionHandlers[i]->initialize(network, ofill);
	}

	// Get the local boundaries
	PetscInt xs, xm, ys, ym;
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	checkPetscError(ierr,
		"PetscSolver2DHandler::createSolverContext: "
		"DMDAGetCorners failed.");
	// Set it in the handler
	setLocalCoordinates(xs, xm, ys, ym);

	// Tell the network the number of grid points on this process with ghosts
	// TODO: do we need the ghost points?
	network.setGridSize(localXM + 2);

	// Get the diagonal fill
	auto nPartials = network.getDiagonalFill(dfill);

	// Load up the block fills
	auto dfillsparse = ConvertToPetscSparseFillMap(dof + 1, dfill);
	auto ofillsparse = ConvertToPetscSparseFillMap(dof + 1, ofill);
	ierr = DMDASetBlockFillsSparse(da, dfillsparse.data(), ofillsparse.data());
	checkPetscError(ierr,
		"PetscSolver2DHandler::createSolverContext: "
		"DMDASetBlockFills failed.");

	// Initialize the arrays for the reaction partial derivatives
	vals = Kokkos::View<double*>("solverPartials", nPartials);

	// Set the size of the partial derivatives vectors
	reactingPartialsForCluster.resize(dof, 0.0);

	return;
}

void
PetscSolver2DHandler::initializeConcentration(DM& da, Vec& C)
{
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar*** concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::initializeConcentration: "
		"DMDAVecGetArrayDOF failed.");

	// Initialize the last temperature at each grid point on this process
	for (auto i = 0; i < localXM + 2; i++) {
		temperature.push_back(0.0);
	}

	// Get the last time step written in the HDF5 file
	bool hasConcentrations = false;
	std::unique_ptr<io::XFile> xfile;
	std::unique_ptr<io::XFile::ConcentrationGroup> concGroup;
	if (not networkName.empty()) {
		xfile.reset(new io::XFile(networkName));
		concGroup = xfile->getGroup<io::XFile::ConcentrationGroup>();
		hasConcentrations = (concGroup and concGroup->hasTimesteps());
	}

	// Give the surface position to the temperature handler
	temperatureHandler->updateSurfacePosition(surfacePosition[0]);

	// Initialize the flux handler
	fluxHandler->initializeFluxHandler(network, surfacePosition[0], grid);

	// Initialize the grid for the diffusion
	diffusionHandler->initializeDiffusionGrid(
		advectionHandlers, grid, localXM, localXS, localYM, hY, localYS);

	// Initialize the grid for the advection
	advectionHandlers[0]->initializeAdvectionGrid(
		advectionHandlers, grid, localXM, localXS, localYM, hY, localYS);

	// Pointer for the concentration vector at a specific grid point
	PetscScalar* concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	// + moments
	const auto dof = network.getDOF();

	// Get the single vacancy ID
	auto singleVacancyCluster = network.getSingleVacancy();
	auto vacancyIndex = NetworkType::invalidIndex();
	if (singleVacancyCluster.getId() != NetworkType::invalidIndex()) {
		vacancyIndex = singleVacancyCluster.getId();
	}

	// Loop on all the grid points
	for (auto j = localYS; j < localYS + localYM; j++)
		for (auto i = (PetscInt)localXS - 1;
			 i <= (PetscInt)localXS + (PetscInt)localXM; i++) {
			// Temperature
			plsm::SpaceVector<double, 3> gridPosition{0.0, j * hY, 0.0};
			if (i < 0)
				gridPosition[0] = (grid[0] - grid[surfacePosition[j] + 1]) /
					(grid[grid.size() - 1] - grid[surfacePosition[j] + 1]);
			else
				gridPosition[0] = ((grid[i] + grid[i + 1]) / 2.0 -
									  grid[surfacePosition[j] + 1]) /
					(grid[grid.size() - 1] - grid[surfacePosition[j] + 1]);
			auto temp = temperatureHandler->getTemperature(gridPosition, 0.0);
			temperature[i - localXS + 1] = temp;

			// Boundary conditions
			if (i < localXS || i >= localXS + localXM)
				continue;

			concOffset = concentrations[j][i];
			concOffset[dof] = temp;

			// Loop on all the clusters to initialize at 0.0
			for (auto n = 0; n < dof; n++) {
				concOffset[n] = 0.0;
			}

			// Initialize the vacancy concentration
			if (i >= surfacePosition[j] + leftOffset and
				vacancyIndex != NetworkType::invalidIndex() and
				not hasConcentrations and i < nX - rightOffset and
				j >= bottomOffset and j < nY - topOffset) {
				concOffset[vacancyIndex] = initialVConc;
			}
		}

	// If the concentration must be set from the HDF5 file
	if (hasConcentrations) {
		assert(concGroup);
		auto tsGroup = concGroup->getLastTimestepGroup();
		assert(tsGroup);

		// Loop on the full grid
		for (auto j = 0; j < nY; j++)
			for (auto i = 0; i < nX; i++) {
				// Read the concentrations from the HDF5 file
				auto concVector = tsGroup->readGridPoint(i, j);

				// Change the concentration only if we are on the locally owned
				// part of the grid
				if (i >= localXS && i < localXS + localXM && j >= localYS &&
					j < localYS + localYM) {
					concOffset = concentrations[j][i];
					// Loop on the concVector size
					for (auto l = 0; l < concVector.size(); l++) {
						concOffset[(IdType)concVector.at(l).at(0)] =
							concVector.at(l).at(1);
					}
					// Get the temperature
					double temp = concVector.at(concVector.size() - 1).at(1);
					temperature[i - localXS + 1] = temp;
				}
			}
	}

	// Update the network with the temperature
	network.setTemperatures(temperature);
	network.syncClusterDataOnHost();

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::initializeConcentration: "
		"DMDAVecRestoreArrayDOF failed.");

	return;
}

void
PetscSolver2DHandler::initGBLocation(DM& da, Vec& C)
{
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar*** concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::initGBLocation: "
		"DMDAVecGetArrayDOF failed.");

	// Pointer for the concentration vector at a specific grid point
	PetscScalar* concOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	// + moments
	const auto dof = network.getDOF();

	// Need to use the NE network here
	using NetworkType = core::network::NEReactionNetwork;
	using Spec = typename NetworkType::Species;
	auto& neNetwork = dynamic_cast<NetworkType&>(network);

	// Loop on the GB
	for (auto const& pair : gbVector) {
		// Get the coordinate of the point
		auto xi = std::get<0>(pair);
		auto yj = std::get<1>(pair);
		// Check if we are on the right process
		if (xi >= localXS && xi < localXS + localXM && yj >= localYS &&
			yj < localYS + localYM) {
			// Get the local concentration
			concOffset = concentrations[yj][xi];

			using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
				Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(concOffset, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);

			// Transfer the local amount of Xe clusters
			setLocalXeRate(
				neNetwork.getTotalAtomConcentration(dConcs, Spec::Xe, 1),
				xi - localXS, yj - localYS);

			// Loop on all the clusters to initialize at 0.0
			for (auto n = 0; n < dof; n++) {
				concOffset[n] = 0.0;
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::initGBLocation: "
		"DMDAVecRestoreArrayDOF failed.");

	return;
}

std::vector<std::vector<std::vector<std::vector<std::pair<IdType, double>>>>>
PetscSolver2DHandler::getConcVector(DM& da, Vec& C)
{
	// Initial declaration
	PetscErrorCode ierr;
	const double* gridPointSolution = nullptr;

	// Pointer for the concentration vector
	PetscScalar*** concentrations = nullptr;
	ierr = DMDAVecGetArrayDOFRead(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::getConcVector: "
		"DMDAVecGetArrayDOFRead failed.");

	// Get the network and dof
	auto& network = getNetwork();
	const auto dof = network.getDOF();

	// Create the vector for the concentrations
	std::vector<
		std::vector<std::vector<std::vector<std::pair<IdType, double>>>>>
		toReturn;
	std::vector<std::vector<std::vector<std::pair<IdType, double>>>>
		tempTempTempVector;

	// Loop on the grid points
	for (auto j = 0; j < localYM; ++j) {
		std::vector<std::vector<std::pair<IdType, double>>> tempTempVector;
		for (auto i = 0; i < localXM; ++i) {
			gridPointSolution = concentrations[localYS + j][localXS + i];

			// Create the temporary vector for this grid point
			std::vector<std::pair<IdType, double>> tempVector;
			for (auto l = 0; l < dof + 1; ++l) {
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

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::getConcVector: "
		"DMDAVecRestoreArrayDOFRead failed.");

	return toReturn;
}

void
PetscSolver2DHandler::setConcVector(DM& da, Vec& C,
	std::vector<
		std::vector<std::vector<std::vector<std::pair<IdType, double>>>>>&
		concVector)
{
	PetscErrorCode ierr;

	// Pointer for the concentration vector
	PetscScalar* gridPointSolution = nullptr;
	PetscScalar*** concentrations = nullptr;
	ierr = DMDAVecGetArrayDOF(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMDAVecGetArrayDOF failed.");

	// Loop on the grid points
	for (auto j = 0; j < localYM; ++j) {
		for (auto i = 0; i < localXM; ++i) {
			gridPointSolution = concentrations[localYS + j][localXS + i];

			// Loop on the given vector
			for (auto l = 0; l < concVector[0][j][i].size(); l++) {
				gridPointSolution[concVector[0][j][i][l].first] =
					concVector[0][j][i][l].second;
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOF(da, C, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMDAVecRestoreArrayDOF failed.");

	// Get the complete data array, including ghost cells to set the temperature
	// at the ghost points
	Vec localSolution;
	ierr = DMGetLocalVector(da, &localSolution);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMGetLocalVector failed.");
	ierr = DMGlobalToLocalBegin(da, C, INSERT_VALUES, localSolution);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMGlobalToLocalBegin failed.");
	ierr = DMGlobalToLocalEnd(da, C, INSERT_VALUES, localSolution);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMGlobalToLocalEnd failed.");
	// Get the array of concentration
	ierr = DMDAVecGetArrayDOFRead(da, localSolution, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMDAVecGetArrayDOFRead failed.");

	// Getthe DOF of the network
	const auto dof = network.getDOF();

	// Loop on the grid points
	for (auto j = 0; j < localYM; ++j) {
		for (auto i = -1; i <= localXM; ++i) {
			gridPointSolution = concentrations[localYS + j][localXS + i];

			// Get the temperature
			temperature[i + 1] = gridPointSolution[dof];
		}
		// Update the network with the temperature
		network.setTemperatures(temperature);
		network.syncClusterDataOnHost();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, localSolution, &concentrations);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMDAVecRestoreArrayDOFRead failed.");
	ierr = DMRestoreLocalVector(da, &localSolution);
	checkPetscError(ierr,
		"PetscSolver2DHandler::setConcVector: "
		"DMRestoreLocalVector failed.");

	return;
}

void
PetscSolver2DHandler::updateConcentration(
	TS& ts, Vec& localC, Vec& F, PetscReal ftime)
{
	PetscErrorCode ierr;

	// Get the local data vector from PETSc
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(
		ierr, "PetscSolver2DHandler::updateConcentration: TSGetDM failed.");

	// Pointers to the PETSc arrays that start at the beginning (localXS,
	// localYS) of the local array
	PetscScalar ***concs = nullptr, ***updatedConcs = nullptr;
	// Get pointers to vector data
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr,
		"PetscSolver2DHandler::updateConcentration: "
		"DMDAVecGetArrayDOFRead (localC) failed.");
	ierr = DMDAVecGetArrayDOF(da, F, &updatedConcs);
	checkPetscError(ierr,
		"PetscSolver2DHandler::updateConcentration: "
		"DMDAVecGetArrayDOF (F) failed.");

	// The following pointers are set to the first position in the conc or
	// updatedConc arrays that correspond to the beginning of the data for the
	// current grid point. They are accessed just like regular arrays.
	PetscScalar *concOffset = nullptr, *updatedConcOffset = nullptr;

	// Degrees of freedom is the total number of clusters in the network
	const auto dof = network.getDOF();

	// Set some step size variable
	double sy = 1.0 / (hY * hY);

	// Declarations for variables used in the loop
	double** concVector = new double*[5];
	plsm::SpaceVector<double, 3> gridPosition{0.0, 0.0, 0.0};
	std::vector<double> incidentFluxVector;
	double atomConc = 0.0, totalAtomConc = 0.0;

	// Loop over grid points first for the temperature, including the ghost
	// points in X
	for (auto yj = localYS; yj < localYS + localYM; yj++) {
		temperatureHandler->updateSurfacePosition(surfacePosition[yj]);
		bool tempHasChanged = false;
		for (auto xi = (PetscInt)localXS - 1;
			 xi <= (PetscInt)localXS + (PetscInt)localXM; xi++) {
			// Heat condition
			if (xi == surfacePosition[yj] && xi >= localXS &&
				xi < localXS + localXM) {
				// Compute the old and new array offsets
				concOffset = concs[yj][xi];
				updatedConcOffset = updatedConcs[yj][xi];

				// Fill the concVector with the pointer to the middle, left, and
				// right grid points
				concVector[0] = concOffset; // middle
				concVector[1] = concs[yj][(PetscInt)xi - 1]; // left
				concVector[2] = concs[yj][xi + 1]; // right
				concVector[3] = concs[(PetscInt)yj - 1][xi]; // bottom
				concVector[4] = concs[yj + 1][xi]; // top

				// Compute the left and right hx
				double hxLeft = 0.0, hxRight = 0.0;
				if (xi >= 1 && xi < nX) {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				}
				else if (xi < 1) {
					hxLeft = grid[xi + 1] - grid[xi];
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				}
				else {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = grid[xi + 1] - grid[xi];
				}

				temperatureHandler->computeTemperature(
					concVector, updatedConcOffset, hxLeft, hxRight, xi, sy, yj);
			}

			// Boundary conditions
			// Everything to the left of the surface is empty
			if (xi < surfacePosition[yj] + leftOffset ||
				xi > nX - 1 - rightOffset) {
				continue;
			}
			// Free surface GB
			bool skip = false;
			for (auto& pair : gbVector) {
				if (xi == std::get<0>(pair) && yj == std::get<1>(pair)) {
					skip = true;
					break;
				}
			}
			if (skip)
				continue;
			// Compute the old and new array offsets
			concOffset = concs[yj][xi];
			updatedConcOffset = updatedConcs[yj][xi];

			// Fill the concVector with the pointer to the middle, left, and
			// right grid points
			concVector[0] = concOffset; // middle
			concVector[1] = concs[yj][(PetscInt)xi - 1]; // left
			concVector[2] = concs[yj][xi + 1]; // right
			concVector[3] = concs[(PetscInt)yj - 1][xi]; // bottom
			concVector[4] = concs[yj + 1][xi]; // top

			// Compute the left and right hx
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < nX) {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else if (xi < 1) {
				hxLeft = grid[xi + 1] - grid[xi];
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = grid[xi + 1] - grid[xi];
			}

			// Set the grid fraction
			gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0 -
								  grid[surfacePosition[yj] + 1]) /
				(grid[grid.size() - 1] - grid[surfacePosition[yj] + 1]);
			gridPosition[1] = yj / nY;

			// Get the temperature from the temperature handler
			temperatureHandler->setTemperature(concOffset);
			double temp =
				temperatureHandler->getTemperature(gridPosition, ftime);

			// Update the network if the temperature changed
			if (std::fabs(temperature[xi + 1 - localXS] - temp) > 0.1) {
				temperature[xi + 1 - localXS] = temp;
				tempHasChanged = true;
			}

			// ---- Compute the temperature over the locally owned part of the
			// grid -----
			if (xi >= localXS && xi < localXS + localXM) {
				temperatureHandler->computeTemperature(
					concVector, updatedConcOffset, hxLeft, hxRight, xi, sy, yj);
			}
		}

		// TODO: it is updated T more than once per MPI process in preparation
		// of T depending on more than X
		if (tempHasChanged) {
			// Update the network with the temperature
			network.setTemperatures(temperature);
			network.syncClusterDataOnHost();
		}
	}

	// Loop over grid points
	for (auto yj = bottomOffset; yj < nY - topOffset; yj++) {
		// Computing the trapped atom concentration is only needed for the
		// attenuation
		if (useAttenuation) {
			// Compute the total concentration of atoms contained in bubbles
			atomConc = 0.0;

			auto& psiNetwork =
				dynamic_cast<core::network::IPSIReactionNetwork&>(network);

			// Loop over grid points
			for (auto xi = surfacePosition[yj] + leftOffset;
				 xi < nX - rightOffset; xi++) {
				// We are only interested in the helium near the surface
				if ((grid[xi] + grid[xi + 1]) / 2.0 -
						grid[surfacePosition[yj] + 1] >
					2.0)
					continue;

				// Check if we are on the right processor
				if (xi >= localXS && xi < localXS + localXM && yj >= localYS &&
					yj < localYS + localYM) {
					// Get the concentrations at this grid point
					concOffset = concs[yj][xi];

					// Sum the total atom concentration
					using HostUnmanaged = Kokkos::View<double*,
						Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
					auto hConcs = HostUnmanaged(concOffset, dof);
					auto dConcs = Kokkos::View<double*>("Concentrations", dof);
					deep_copy(dConcs, hConcs);
					atomConc += psiNetwork.getTotalTrappedHeliumConcentration(
									dConcs, 0) *
						(grid[xi + 1] - grid[xi]);
				}
			}

			// Share the concentration with all the processes
			totalAtomConc = 0.0;
			auto xolotlComm = util::getMPIComm();
			MPI_Allreduce(
				&atomConc, &totalAtomConc, 1, MPI_DOUBLE, MPI_SUM, xolotlComm);

			// Set the disappearing rate in the modified TM handler
			psiNetwork.updateTrapMutationDisappearingRate(totalAtomConc);
		}

		// Skip if we are not on the right process
		if (yj < localYS || yj >= localYS + localYM)
			continue;

		// Set the grid position
		gridPosition[1] = yj * hY;

		// Initialize the flux, advection handlers which depend
		// on the surface position at Y
		fluxHandler->initializeFluxHandler(network, surfacePosition[yj], grid);
		advectionHandlers[0]->setLocation(
			grid[surfacePosition[yj] + 1] - grid[1]);

		for (auto xi = localXS; xi < localXS + localXM; xi++) {
			// Compute the old and new array offsets
			concOffset = concs[yj][xi];
			updatedConcOffset = updatedConcs[yj][xi];

			// Fill the concVector with the pointer to the middle, left, right,
			// bottom, and top grid points
			concVector[0] = concOffset; // middle
			concVector[1] = concs[yj][(PetscInt)xi - 1]; // left
			concVector[2] = concs[yj][xi + 1]; // right
			concVector[3] = concs[(PetscInt)yj - 1][xi]; // bottom
			concVector[4] = concs[yj + 1][xi]; // top

			// Compute the left and right hx
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < nX) {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else if (xi < 1) {
				hxLeft = grid[xi + 1] - grid[xi];
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = grid[xi + 1] - grid[xi];
			}

			// Boundary conditions
			// Everything to the left of the surface is empty
			if (xi < surfacePosition[yj] + leftOffset ||
				xi > nX - 1 - rightOffset || yj < bottomOffset ||
				yj > nY - 1 - topOffset) {
				continue;
			}
			// Free surface GB
			bool skip = false;
			for (auto& pair : gbVector) {
				if (xi == std::get<0>(pair) && yj == std::get<1>(pair)) {
					skip = true;
					break;
				}
			}
			if (skip)
				continue;

			// ----- Account for flux of incoming particles -----
			fluxHandler->computeIncidentFlux(
				ftime, updatedConcOffset, xi, surfacePosition[yj]);

			// ---- Compute diffusion over the locally owned part of the grid
			// -----
			diffusionHandler->computeDiffusion(network, concVector,
				updatedConcOffset, hxLeft, hxRight, xi - localXS, sy,
				yj - localYS);

			// ---- Compute advection over the locally owned part of the grid
			// ----- Set the grid position
			gridPosition[0] = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
			for (auto i = 0; i < advectionHandlers.size(); i++) {
				advectionHandlers[i]->computeAdvection(network, gridPosition,
					concVector, updatedConcOffset, hxLeft, hxRight,
					xi - localXS, hY, yj - localYS);
			}

			auto surfacePos = grid[surfacePosition[yj] + 1];
			auto curXPos = (grid[xi] + grid[xi + 1]) / 2.0;
			auto prevXPos = (grid[xi - 1] + grid[xi]) / 2.0;
			auto curDepth = curXPos - surfacePos;
			auto curSpacing = curXPos - prevXPos;

			// ----- Compute the reaction fluxes over the locally owned part of
			// the grid -----
			using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
				Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(concOffset, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);
			auto hFlux = HostUnmanaged(updatedConcOffset, dof);
			auto dFlux = Kokkos::View<double*>("Fluxes", dof);
			deep_copy(dFlux, hFlux);
			fluxCounter->increment();
			fluxTimer->start();
			network.computeAllFluxes(
				dConcs, dFlux, xi + 1 - localXS, curDepth, curSpacing);
			fluxTimer->stop();
			deep_copy(hFlux, dFlux);
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr,
		"PetscSolver2DHandler::updateConcentration: "
		"DMDAVecRestoreArrayDOFRead (localC) failed.");
	ierr = DMDAVecRestoreArrayDOF(da, F, &updatedConcs);
	checkPetscError(ierr,
		"PetscSolver2DHandler::updateConcentration: "
		"DMDAVecRestoreArrayDOF (F) failed.");

	// Clear memory
	delete[] concVector;

	return;
}

void
PetscSolver2DHandler::computeJacobian(
	TS& ts, Vec& localC, Mat& J, PetscReal ftime)
{
	PetscErrorCode ierr;

	// Get the distributed array
	DM da;
	ierr = TSGetDM(ts, &da);
	checkPetscError(ierr,
		"PetscSolver2DHandler::computeJacobian: "
		"TSGetDM failed.");

	// Get pointers to vector data
	PetscScalar*** concs = nullptr;
	ierr = DMDAVecGetArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr,
		"PetscSolver2DHandler::computeJacobian: "
		"DMDAVecGetArrayDOFRead failed.");

	// The degree of freedom is the size of the network
	const auto dof = network.getDOF();

	// Setup some step size variables
	double sy = 1.0 / (hY * hY);

	// Pointer to the concentrations at a given grid point
	PetscScalar* concOffset = nullptr;

	// Arguments for MatSetValuesStencil called below
	MatStencil rowId;
	MatStencil colIds[dof];
	IdType pdColIdsVectorSize = 0;

	// Declarations for variables used in the loop
	double atomConc = 0.0, totalAtomConc = 0.0;
	plsm::SpaceVector<double, 3> gridPosition{0.0, 0.0, 0.0};

	// Get the total number of diffusing clusters
	const auto nDiff = std::max(diffusionHandler->getNumberOfDiffusing(), 0);

	// Get the total number of advecting clusters
	int nAdvec = 0;
	for (auto l = 0; l < advectionHandlers.size(); l++) {
		auto n = advectionHandlers[l]->getNumberOfAdvecting();
		if (n > nAdvec)
			nAdvec = n;
	}

	// Arguments for MatSetValuesStencil called below
	MatStencil row, cols[5];
	PetscScalar tempVals[5];
	IdType tempIndices[1];
	PetscScalar diffVals[5 * nDiff];
	IdType diffIndices[nDiff];
	PetscScalar advecVals[2 * nAdvec];
	IdType advecIndices[nAdvec];

	/*
	 Loop over grid points for the temperature, including ghosts
	 */
	for (auto yj = localYS; yj < localYS + localYM; yj++) {
		temperatureHandler->updateSurfacePosition(surfacePosition[yj]);
		bool tempHasChanged = false;
		for (auto xi = (PetscInt)localXS - 1;
			 xi <= (PetscInt)localXS + (PetscInt)localXM; xi++) {
			// Compute the left and right hx
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < nX) {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else if (xi == -1) {
				hxLeft = 0.0;
				hxRight = (grid[xi + 2] + grid[xi + 1]) / 2.0;
			}
			else if (xi < 1) {
				hxLeft = grid[xi + 1] - grid[xi];
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = grid[xi + 1] - grid[xi];
			}

			// Heat condition
			if (xi == surfacePosition[yj] && xi >= localXS &&
				xi < localXS + localXM) {
				// Get the partial derivatives for the temperature
				auto setValues =
					temperatureHandler->computePartialsForTemperature(
						tempVals, tempIndices, hxLeft, hxRight, xi, sy, yj);

				if (setValues) {
					// Set grid coordinate and component number for the row
					row.i = xi;
					row.j = yj;
					row.c = tempIndices[0];

					// Set grid coordinates and component numbers for the
					// columns corresponding to the middle, left, and right grid
					// points
					cols[0].i = xi; // middle
					cols[0].j = yj;
					cols[0].c = tempIndices[0];
					cols[1].i = (PetscInt)xi - 1; // left
					cols[1].j = yj;
					cols[1].c = tempIndices[0];
					cols[2].i = xi + 1; // right
					cols[2].j = yj;
					cols[2].c = tempIndices[0];
					cols[3].i = xi; // bottom
					cols[3].j = (PetscInt)yj - 1;
					cols[3].c = tempIndices[0];
					cols[4].i = xi; // top
					cols[4].j = yj + 1;
					cols[4].c = tempIndices[0];

					ierr = MatSetValuesStencil(
						J, 1, &row, 5, cols, tempVals, ADD_VALUES);
					checkPetscError(ierr,
						"PetscSolver2DHandler::computeJacobian: "
						"MatSetValuesStencil (temperature) failed.");
				}
			}

			// Boundary conditions
			// Everything to the left of the surface is empty
			if (xi < surfacePosition[yj] + leftOffset ||
				xi > nX - 1 - rightOffset)
				continue;
			// Free surface GB
			bool skip = false;
			for (auto& pair : gbVector) {
				if (xi == std::get<0>(pair) && yj == std::get<1>(pair)) {
					skip = true;
					break;
				}
			}
			if (skip)
				continue;

			// Get the concentrations at this grid point
			concOffset = concs[yj][xi];

			// Set the grid fraction
			gridPosition[0] = ((grid[xi] + grid[xi + 1]) / 2.0 -
								  grid[surfacePosition[yj] + 1]) /
				(grid[grid.size() - 1] - grid[surfacePosition[yj] + 1]);
			gridPosition[1] = yj / nY;

			// Get the temperature from the temperature handler
			temperatureHandler->setTemperature(concOffset);
			double temp =
				temperatureHandler->getTemperature(gridPosition, ftime);

			// Update the network if the temperature changed
			if (std::fabs(temperature[xi + 1 - localXS] - temp) > 0.1) {
				temperature[xi + 1 - localXS] = temp;
				tempHasChanged = true;
			}

			// Get the partial derivatives for the temperature
			if (xi >= localXS && xi < localXS + localXM) {
				auto setValues =
					temperatureHandler->computePartialsForTemperature(
						tempVals, tempIndices, hxLeft, hxRight, xi, sy, yj);

				if (setValues) {
					// Set grid coordinate and component number for the row
					row.i = xi;
					row.j = yj;
					row.c = tempIndices[0];

					// Set grid coordinates and component numbers for the
					// columns corresponding to the middle, left, and right grid
					// points
					cols[0].i = xi; // middle
					cols[0].j = yj;
					cols[0].c = tempIndices[0];
					cols[1].i = (PetscInt)xi - 1; // left
					cols[1].j = yj;
					cols[1].c = tempIndices[0];
					cols[2].i = xi + 1; // right
					cols[2].j = yj;
					cols[2].c = tempIndices[0];
					cols[3].i = xi; // bottom
					cols[3].j = (PetscInt)yj - 1;
					cols[3].c = tempIndices[0];
					cols[4].i = xi; // top
					cols[4].j = yj + 1;
					cols[4].c = tempIndices[0];

					ierr = MatSetValuesStencil(
						J, 1, &row, 5, cols, tempVals, ADD_VALUES);
					checkPetscError(ierr,
						"PetscSolver2DHandler::computeJacobian: "
						"MatSetValuesStencil (temperature) failed.");
				}
			}
		}

		if (tempHasChanged) {
			// Update the network with the temperature
			network.setTemperatures(temperature);
			network.syncClusterDataOnHost();
		}
	}

	// Loop over the grid points
	for (auto yj = bottomOffset; yj < nY - topOffset; yj++) {
		// Computing the trapped atom concentration is only needed for the
		// attenuation
		if (useAttenuation) {
			// Compute the total concentration of atoms contained in bubbles
			atomConc = 0.0;

			auto& psiNetwork =
				dynamic_cast<core::network::IPSIReactionNetwork&>(network);

			// Loop over grid points
			for (auto xi = surfacePosition[yj] + leftOffset;
				 xi < nX - rightOffset; xi++) {
				// We are only interested in the helium near the surface
				if ((grid[xi] + grid[xi + 1]) / 2.0 -
						grid[surfacePosition[yj] + 1] >
					2.0)
					continue;

				// Check if we are on the right processor
				if (xi >= localXS && xi < localXS + localXM && yj >= localYS &&
					yj < localYS + localYM) {
					// Get the concentrations at this grid point
					concOffset = concs[yj][xi];

					// Sum the total atom concentration
					using HostUnmanaged = Kokkos::View<double*,
						Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
					auto hConcs = HostUnmanaged(concOffset, dof);
					auto dConcs = Kokkos::View<double*>("Concentrations", dof);
					deep_copy(dConcs, hConcs);
					atomConc += psiNetwork.getTotalTrappedHeliumConcentration(
									dConcs, 0) *
						(grid[xi + 1] - grid[xi]);
				}
			}

			// Share the concentration with all the processes
			totalAtomConc = 0.0;
			auto xolotlComm = util::getMPIComm();
			MPI_Allreduce(
				&atomConc, &totalAtomConc, 1, MPI_DOUBLE, MPI_SUM, xolotlComm);

			// Set the disappearing rate in the modified TM handler
			psiNetwork.updateTrapMutationDisappearingRate(totalAtomConc);
		}

		// Skip if we are not on the right process
		if (yj < localYS || yj >= localYS + localYM)
			continue;

		// Set the grid position
		gridPosition[1] = yj * hY;

		// Initialize the advection and temperature handlers which depend
		// on the surface position at Y
		advectionHandlers[0]->setLocation(
			grid[surfacePosition[yj] + 1] - grid[1]);

		for (auto xi = localXS; xi < localXS + localXM; xi++) {
			// Boundary conditions
			// Everything to the left of the surface is empty
			if (xi < surfacePosition[yj] + leftOffset ||
				xi > nX - 1 - rightOffset || yj < bottomOffset ||
				yj > nY - 1 - topOffset)
				continue;
			// Free surface GB
			bool skip = false;
			for (auto& pair : gbVector) {
				if (xi == std::get<0>(pair) && yj == std::get<1>(pair)) {
					skip = true;
					break;
				}
			}
			if (skip)
				continue;

			// Compute the left and right hx
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < nX) {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else if (xi < 1) {
				hxLeft = grid[xi + 1] - grid[xi];
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = grid[xi + 1] - grid[xi];
			}

			// Get the partial derivatives for the diffusion
			diffusionHandler->computePartialsForDiffusion(network, diffVals,
				diffIndices, hxLeft, hxRight, xi - localXS, sy, yj - localYS);

			// Loop on the number of diffusion cluster to set the values in the
			// Jacobian
			for (auto i = 0; i < nDiff; i++) {
				// Set grid coordinate and component number for the row
				row.i = xi;
				row.j = yj;
				row.c = diffIndices[i];

				// Set grid coordinates and component numbers for the columns
				// corresponding to the middle, left, right, bottom, and top
				// grid points
				cols[0].i = xi; // middle
				cols[0].j = yj;
				cols[0].c = diffIndices[i];
				cols[1].i = (PetscInt)xi - 1; // left
				cols[1].j = yj;
				cols[1].c = diffIndices[i];
				cols[2].i = xi + 1; // right
				cols[2].j = yj;
				cols[2].c = diffIndices[i];
				cols[3].i = xi; // bottom
				cols[3].j = (PetscInt)yj - 1;
				cols[3].c = diffIndices[i];
				cols[4].i = xi; // top
				cols[4].j = yj + 1;
				cols[4].c = diffIndices[i];

				ierr = MatSetValuesStencil(
					J, 1, &row, 5, cols, diffVals + (5 * i), ADD_VALUES);
				checkPetscError(ierr,
					"PetscSolver2DHandler::computeJacobian: "
					"MatSetValuesStencil (diffusion) failed.");
			}

			// Get the partial derivatives for the advection
			// Set the grid position
			gridPosition[0] = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
			for (auto l = 0; l < advectionHandlers.size(); l++) {
				advectionHandlers[l]->computePartialsForAdvection(network,
					advecVals, advecIndices, gridPosition, hxLeft, hxRight,
					xi - localXS, hY, yj - localYS);

				// Get the stencil indices to know where to put the partial
				// derivatives in the Jacobian
				auto advecStencil =
					advectionHandlers[l]->getStencilForAdvection(gridPosition);

				// Get the number of advecting clusters
				nAdvec = advectionHandlers[l]->getNumberOfAdvecting();

				// Loop on the number of advecting cluster to set the values in
				// the Jacobian
				for (auto i = 0; i < nAdvec; i++) {
					// Set grid coordinate and component number for the row
					row.i = xi;
					row.j = yj;
					row.c = advecIndices[i];

					// If we are on the sink, the partial derivatives are not
					// the same Both sides are giving their concentrations to
					// the center
					if (advectionHandlers[l]->isPointOnSink(gridPosition)) {
						cols[0].i = (PetscInt)xi - advecStencil[0]; // left?
						cols[0].j = yj - advecStencil[1]; // bottom?
						cols[0].c = advecIndices[i];
						cols[1].i = xi + advecStencil[0]; // right?
						cols[1].j = yj + advecStencil[1]; // top?
						cols[1].c = advecIndices[i];
					}
					else {
						// Set grid coordinates and component numbers for the
						// columns corresponding to the middle and other grid
						// points
						cols[0].i = xi; // middle
						cols[0].j = yj;
						cols[0].c = advecIndices[i];
						cols[1].i = xi + advecStencil[0]; // left or right?
						cols[1].j = yj + advecStencil[1]; // bottom or top?
						cols[1].c = advecIndices[i];
					}

					// Update the matrix
					ierr = MatSetValuesStencil(
						J, 1, &row, 2, cols, advecVals + (2 * i), ADD_VALUES);
					checkPetscError(ierr,
						"PetscSolver2DHandler::computeJacobian: "
						"MatSetValuesStencil (advection) failed.");
				}
			}

			// Get the concentations
			concOffset = concs[yj][xi];

			// ----- Take care of the reactions for all the reactants -----

			auto surfacePos = grid[surfacePosition[yj] + 1];
			auto curXPos = (grid[xi] + grid[xi + 1]) / 2.0;
			auto prevXPos = (grid[xi - 1] + grid[xi]) / 2.0;
			auto curDepth = curXPos - surfacePos;
			auto curSpacing = curXPos - prevXPos;

			// Compute all the partial derivatives for the reactions
			using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
				Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(concOffset, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);
			partialDerivativeCounter->increment();
			partialDerivativeTimer->start();
			network.computeAllPartials(
				dConcs, vals, xi + 1 - localXS, curDepth, curSpacing);
			partialDerivativeTimer->stop();
			auto hPartials = create_mirror_view(vals);
			deep_copy(hPartials, vals);

			// Variable for the loop on reactants
			IdType startingIdx = 0;
			// Update the column in the Jacobian that represents each DOF
			for (auto i = 0; i < dof; i++) {
				// Set grid coordinate and component number for the row
				rowId.i = xi;
				rowId.j = yj;
				rowId.c = i;

				// Number of partial derivatives
				auto rowIter = dfill.find(i);
				if (rowIter != dfill.end()) {
					const auto& row = rowIter->second;
					pdColIdsVectorSize = row.size();

					// Loop over the list of column ids
					for (auto j = 0; j < pdColIdsVectorSize; j++) {
						// Set grid coordinate and component number for a column
						// in the list
						colIds[j].i = xi;
						colIds[j].j = yj;
						colIds[j].c = row[j];
						// Get the partial derivative from the array of all of
						// the partials
						reactingPartialsForCluster[j] =
							hPartials(startingIdx + j);
					}
					// Update the matrix
					ierr = MatSetValuesStencil(J, 1, &rowId, pdColIdsVectorSize,
						colIds, reactingPartialsForCluster.data(), ADD_VALUES);
					checkPetscError(ierr,
						"PetscSolver2DHandler::computeJacobian: "
						"MatSetValuesStencil (reactions) failed.");

					// Increase the starting index
					startingIdx += pdColIdsVectorSize;
				}
			}
		}
	}

	/*
	 Restore vectors
	 */
	ierr = DMDAVecRestoreArrayDOFRead(da, localC, &concs);
	checkPetscError(ierr,
		"PetscSolver2DHandler::computeJacobian: "
		"DMDAVecRestoreArrayDOFRead failed.");

	return;
}

} /* end namespace handler */
} /* end namespace solver */
} /* end namespace xolotl */
