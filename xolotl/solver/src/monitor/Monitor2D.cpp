// Includes
#include <petscsys.h>
#include <petscts.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <xolotl/core/Constants.h>
#include <xolotl/core/network/NEReactionNetwork.h>
#include <xolotl/core/network/PSIReactionNetwork.h>
#include <xolotl/io/XFile.h>
#include <xolotl/perf/xolotlPerf.h>
#include <xolotl/solver/PetscSolver.h>
#include <xolotl/solver/monitor/Monitor.h>
#include <xolotl/util/MPIUtils.h>
#include <xolotl/util/MathUtils.h>
#include <xolotl/util/RandomNumberGenerator.h>
#include <xolotl/viz/IPlot.h>
#include <xolotl/viz/LabelProvider.h>
#include <xolotl/viz/PlotType.h>
#include <xolotl/viz/VizHandlerRegistry.h>
#include <xolotl/viz/dataprovider/CvsXDataProvider.h>
#include <xolotl/viz/dataprovider/CvsXYDataProvider.h>

namespace xolotl
{
namespace solver
{
namespace monitor
{
// Declaration of the functions defined in Monitor.cpp
extern PetscErrorCode
checkTimeStep(TS ts);
extern PetscErrorCode
monitorTime(TS ts, PetscInt timestep, PetscReal time, Vec solution, void* ictx);
extern PetscErrorCode
computeFluence(
	TS ts, PetscInt timestep, PetscReal time, Vec solution, void* ictx);
extern PetscErrorCode
monitorPerf(TS ts, PetscInt timestep, PetscReal time, Vec solution, void* ictx);

// Declaration of the variables defined in Monitor.cpp
extern std::shared_ptr<viz::IPlot> perfPlot;
extern double timeStepThreshold;

//! How often HDF5 file is written
PetscReal hdf5Stride2D = 0.0;
//! Previous time for HDF5
PetscInt hdf5Previous2D = 0;
//! HDF5 output file name
std::string hdf5OutputName2D = "xolotlStop.h5";
//! The pointer to the 2D plot used in MonitorSurface.
std::shared_ptr<viz::IPlot> surfacePlot2D;
//! The variable to store the interstitial flux at the previous time step.
std::vector<double> previousIFlux2D;
//! The variable to store the total number of interstitials going through the
//! surface.
std::vector<double> nInterstitial2D;
//! The variable to store the helium flux at the previous time step.
std::vector<double> previousHeFlux2D;
//! The variable to store the total number of helium going through the bottom.
std::vector<double> nHelium2D;
//! The variable to store the deuterium flux at the previous time step.
std::vector<double> previousDFlux2D;
//! The variable to store the total number of deuterium going through the
//! bottom.
std::vector<double> nDeuterium2D;
//! The variable to store the tritium flux at the previous time step.
std::vector<double> previousTFlux2D;
//! The variable to store the total number of tritium going through the bottom.
std::vector<double> nTritium2D;
//! The variable to store the sputtering yield at the surface.
double sputteringYield2D = 0.0;
// The vector of depths at which bursting happens
std::vector<std::pair<int, int>> depthPositions2D;
// The vector of ids for diffusing interstitial clusters
std::vector<int> iClusterIds2D;
// The id of the largest cluster
int largestClusterId2D = -1;
// The concentration threshold for the largest cluster
double largestThreshold2D = 1.0e-12;

// Timers
std::shared_ptr<perf::ITimer> gbTimer;

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorLargest2D")
/**
 * This is a monitoring method that looks at the largest cluster concentration
 */
PetscErrorCode
monitorLargest2D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	// Initial declaration
	PetscErrorCode ierr;
	double ***solutionArray, *gridPointSolution;
	PetscInt xs, xm, ys, ym;

	PetscFunctionBeginUser;

	// Get the MPI communicator
	auto xolotlComm = util::getMPIComm();
	// Get the number of processes
	int worldSize;
	MPI_Comm_size(xolotlComm, &worldSize);
	// Gets the process ID (important when it is running in parallel)
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);

	// Loop on the local grid
	for (PetscInt j = ys; j < ys + ym; j++)
		for (PetscInt i = xs; i < xs + xm; i++) {
			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[j][i];
			// Check the concentration
			if (gridPointSolution[largestClusterId2D] > largestThreshold2D) {
				ierr = TSSetConvergedReason(ts, TS_CONVERGED_USER);
				CHKERRQ(ierr);
				// Send an error
				throw std::string("\nxolotlSolver::Monitor2D: The largest "
								  "cluster concentration is too high!!");
			}
		}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "startStop2D")
/**
 * This is a monitoring method that will update an hdf5 file at each time step.
 */
PetscErrorCode
startStop2D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	// Initial declaration
	PetscErrorCode ierr;
	const double ***solutionArray, *gridPointSolution;
	PetscInt xs, xm, Mx, ys, ym, My;

	PetscFunctionBeginUser;

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Compute the dt
	double previousTime = solverHandler.getPreviousTime();
	double dt = time - previousTime;

	// Don't do anything if it is not on the stride
	if (((int)((time + dt / 10.0) / hdf5Stride2D) <= hdf5Previous2D) &&
		timestep > 0)
		PetscFunctionReturn(0);

	// Update the previous time
	if ((int)((time + dt / 10.0) / hdf5Stride2D) > hdf5Previous2D)
		hdf5Previous2D++;

	// Gets the process ID (important when it is running in parallel)
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);
	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the network and dof
	auto& network = solverHandler.getNetwork();
	const int dof = network.getDOF();

	// Create an array for the concentration
	double concArray[dof + 1][2];

	// Get the vector of positions of the surface
	std::vector<int> surfaceIndices;
	for (PetscInt i = 0; i < My; i++) {
		surfaceIndices.push_back(solverHandler.getSurfacePosition(i));
	}

	// Open the existing checkpoint file.
	io::XFile checkpointFile(
		hdf5OutputName2D, xolotlComm, io::XFile::AccessMode::OpenReadWrite);

	// Get the current time step
	double currentTimeStep;
	ierr = TSGetTimeStep(ts, &currentTimeStep);
	CHKERRQ(ierr);

	// Add a concentration sub group
	auto concGroup = checkpointFile.getGroup<io::XFile::ConcentrationGroup>();
	assert(concGroup);
	auto tsGroup = concGroup->addTimestepGroup(
		timestep, time, previousTime, currentTimeStep);

	if (solverHandler.moveSurface()) {
		// Write the surface positions and the associated interstitial
		// quantities in the concentration sub group
		tsGroup->writeSurface2D(
			surfaceIndices, nInterstitial2D, previousIFlux2D);
	}

	// Write the bottom impurity information if the bottom is a free surface
	if (solverHandler.getRightOffset() == 1)
		tsGroup->writeBottom2D(nHelium2D, previousHeFlux2D, nDeuterium2D,
			previousDFlux2D, nTritium2D, previousTFlux2D);

	// Loop on the full grid
	for (PetscInt j = 0; j < My; j++) {
		for (PetscInt i = 0; i < Mx; i++) {
			// Wait for all the processes
			MPI_Barrier(xolotlComm);

			// Size of the concentration that will be stored
			int concSize = -1;
			// To save which proc has the information
			int concId = 0;
			// To know which process should write
			bool write = false;

			// If it is the locally owned part of the grid
			if (i >= xs && i < xs + xm && j >= ys && j < ys + ym) {
				write = true;
				// Get the pointer to the beginning of the solution data for
				// this grid point
				gridPointSolution = solutionArray[j][i];

				// Loop on the concentrations
				for (int l = 0; l < dof + 1; l++) {
					if (std::fabs(gridPointSolution[l]) > 1.0e-16) {
						// Increase concSize
						concSize++;
						// Fill the concArray
						concArray[concSize][0] = (double)l;
						concArray[concSize][1] = gridPointSolution[l];
					}
				}

				// Increase concSize one last time
				concSize++;

				// Save the procId
				concId = procId;
			}

			// Get which processor will send the information
			int concProc = 0;
			MPI_Allreduce(&concId, &concProc, 1, MPI_INT, MPI_SUM, xolotlComm);

			// Broadcast the size
			MPI_Bcast(&concSize, 1, MPI_INT, concProc, xolotlComm);

			// Skip the grid point if the size is 0
			if (concSize == 0)
				continue;

			// All processes create the dataset and fill it
			tsGroup->writeConcentrationDataset(
				concSize, concArray, write, i, j);
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeHeliumRetention2D")
/**
 * This is a monitoring method that will compute the helium retention
 */
PetscErrorCode
computeHeliumRetention2D(TS ts, PetscInt, PetscReal time, Vec solution, void*)
{
	// Initial declarations
	PetscErrorCode ierr;
	PetscInt xs, xm, ys, ym, Mx, My;

	PetscFunctionBeginUser;

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the flux handler.
	auto fluxHandler = solverHandler.getFluxHandler();
	// Get the diffusion handler
	auto diffusionHandler = solverHandler.getDiffusionHandler();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);
	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the physical grid in the x direction
	auto grid = solverHandler.getXGrid();

	// Setup step size variables
	double hy = solverHandler.getStepSizeY();

	// Get the network
	using NetworkType =
		core::network::PSIReactionNetwork<core::network::PSIFullSpeciesList>;
	using Spec = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	const int dof = network.getDOF();

	// Get the array of concentration
	double ***solutionArray, *gridPointSolution;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Store the concentration over the grid
	double heConcentration = 0.0, dConcentration = 0.0, tConcentration = 0.0;

	// Loop on the grid
	for (PetscInt yj = ys; yj < ys + ym; yj++) {
		// Get the surface position
		int surfacePos = solverHandler.getSurfacePosition(yj);

		for (PetscInt xi = xs; xi < xs + xm; xi++) {
			// Boundary conditions
			if (xi < surfacePos + solverHandler.getLeftOffset() ||
				xi >= Mx - solverHandler.getRightOffset())
				continue;

			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[yj][xi];

			double hx = grid[xi + 1] - grid[xi];

			using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
				Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(gridPointSolution, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);

			// Get the total concentrations at this grid point
			heConcentration +=
				network.getTotalAtomConcentration(dConcs, Spec::He, 1) * hx *
				hy;
			dConcentration +=
				network.getTotalAtomConcentration(dConcs, Spec::D, 1) * hx * hy;
			tConcentration +=
				network.getTotalAtomConcentration(dConcs, Spec::T, 1) * hx * hy;
		}
	}

	// Get the current process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Sum all the concentrations through MPI reduce
	std::array<double, 3> myConcData{
		heConcentration, dConcentration, tConcentration};
	std::array<double, 3> totalConcData{0.0, 0.0, 0.0};

	MPI_Reduce(myConcData.data(), totalConcData.data(), myConcData.size(),
		MPI_DOUBLE, MPI_SUM, 0, xolotlComm);

	// Extract total He, D, T concentrations.  Values are valid only on rank 0.
	double totalHeConcentration = totalConcData[0];
	double totalDConcentration = totalConcData[1];
	double totalTConcentration = totalConcData[2];

	// Get the total size of the grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Look at the fluxes going in the bulk if the bottom is a free surface
	if (solverHandler.getRightOffset() == 1) {
		// Set the bottom surface position
		int xi = Mx - 2;

		// Get the vector of diffusing clusters
		auto diffusingIds = diffusionHandler->getDiffusingIds();

		// Loop on every Y position
		for (PetscInt j = 0; j < My; j++) {
			// Value to know on which processor is the bottom
			int bottomProc = 0;

			// Check we are on the right proc
			if (xi >= xs && xi < xs + xm && j >= ys && j < ys + ym) {
				// Get the delta time from the previous timestep to this
				// timestep
				double dt = time - solverHandler.getPreviousTime();
				// Compute the total number of impurities that went in the bulk
				nHelium2D[j] += previousHeFlux2D[j] * dt;
				nDeuterium2D[j] += previousDFlux2D[j] * dt;
				nTritium2D[j] += previousTFlux2D[j] * dt;
				previousHeFlux2D[j] = 0.0;
				previousDFlux2D[j] = 0.0;
				previousTFlux2D[j] = 0.0;

				// Get the pointer to the beginning of the solution data for
				// this grid point
				gridPointSolution = solutionArray[j][xi];

				// Factor for finite difference
				double hxLeft = 0.0, hxRight = 0.0;
				if (xi - 1 >= 0 && xi < Mx) {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				}
				else if (xi - 1 < 0) {
					hxLeft = grid[xi + 1] - grid[xi];
					hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
				}
				else {
					hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
					hxRight = grid[xi + 1] - grid[xi];
				}
				double factor = 2.0 * hy / (hxLeft + hxRight);

				// Loop on the diffusing clusters
				for (auto l : diffusingIds) {
					// Get the cluster and composition
					auto cluster = network.getClusterCommon(l);
					auto reg = network.getCluster(l, plsm::onHost).getRegion();
					Composition comp = reg.getOrigin();
					// Get its concentration
					double conc = gridPointSolution[l];
					// Get its size and diffusion coefficient
					int size = comp[Spec::He] + comp[Spec::D] + comp[Spec::T];
					double coef = cluster.getDiffusionCoefficient(xi - xs);
					// Compute the flux going to the right
					double newFlux =
						(double)size * factor * coef * conc * hxRight;
					if (comp.isOnAxis(Spec::He))
						previousHeFlux2D[j] += newFlux;
					if (comp.isOnAxis(Spec::D))
						previousDFlux2D[j] += newFlux;
					if (comp.isOnAxis(Spec::T))
						previousTFlux2D[j] += newFlux;
				}

				// Set the bottom processor
				bottomProc = procId;
			}

			// Get which processor will send the information
			int bottomId = 0;
			MPI_Allreduce(
				&bottomProc, &bottomId, 1, MPI_INT, MPI_SUM, xolotlComm);

			// Send the information about impurities
			// to the other processes
			std::array<double, 6> countFluxData{nHelium2D[j],
				previousHeFlux2D[j], nDeuterium2D[j], previousDFlux2D[j],
				nTritium2D[j], previousTFlux2D[j]};
			MPI_Bcast(countFluxData.data(), countFluxData.size(), MPI_DOUBLE,
				bottomId, xolotlComm);

			// Extract inpurity data from broadcast buffer.
			nHelium2D[j] = countFluxData[0];
			previousHeFlux2D[j] = countFluxData[1];
			nDeuterium2D[j] = countFluxData[2];
			previousDFlux2D[j] = countFluxData[3];
			nTritium2D[j] = countFluxData[4];
			previousTFlux2D[j] = countFluxData[5];
		}
	}

	// Master process
	if (procId == 0) {
		// Compute the total surface irradiated by the helium flux
		double surface = (double)My * hy;

		// Rescale the concentration
		totalHeConcentration = totalHeConcentration / surface;
		totalDConcentration = totalDConcentration / surface;
		totalTConcentration = totalTConcentration / surface;
		double totalHeBulk = 0.0, totalDBulk = 0.0, totalTBulk = 0.0;
		// Look if the bottom is a free surface
		if (solverHandler.getRightOffset() == 1) {
			for (int i = 0; i < My; i++) {
				totalHeBulk += nHelium2D[i];
				totalDBulk += nDeuterium2D[i];
				totalTBulk += nTritium2D[i];
			}
			totalHeBulk = totalHeBulk / surface;
			totalDBulk = totalDBulk / surface;
			totalTBulk = totalTBulk / surface;
		}

		// Get the fluence
		double fluence = fluxHandler->getFluence();

		// Print the result
		std::cout << "\nTime: " << time << std::endl;
		std::cout << "Helium content = " << totalHeConcentration << std::endl;
		std::cout << "Deuterium content = " << totalDConcentration << std::endl;
		std::cout << "Tritium content = " << totalTConcentration << std::endl;
		std::cout << "Fluence = " << fluence << "\n" << std::endl;

		// Uncomment to write the retention and the fluence in a file
		std::ofstream outputFile;
		outputFile.open("retentionOut.txt", std::ios::app);
		outputFile << fluence << " " << totalHeConcentration << " "
				   << totalDConcentration << " " << totalTConcentration << " "
				   << totalHeBulk << " " << totalDBulk << " " << totalTBulk
				   << std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeXenonRetention2D")
/**
 * This is a monitoring method that will compute the xenon retention
 */
PetscErrorCode
computeXenonRetention2D(
	TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(gbTimer);

	// Initial declarations
	PetscErrorCode ierr;
	PetscInt xs, xm, ys, ym;

	PetscFunctionBeginUser;

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);

	// Get the total size of the grid
	PetscInt Mx, My;
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// Setup step size variables
	double hy = solverHandler.getStepSizeY();

	using NetworkType = core::network::NEReactionNetwork;
	using Spec = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;

	// Degrees of freedom is the total number of clusters in the network
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	const int dof = network.getDOF();

	// Get the complete data array, including ghost cells
	Vec localSolution;
	ierr = DMGetLocalVector(da, &localSolution);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, solution, INSERT_VALUES, localSolution);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, solution, INSERT_VALUES, localSolution);
	CHKERRQ(ierr);
	// Get the array of concentration
	PetscReal ***solutionArray, *gridPointSolution;
	ierr = DMDAVecGetArrayDOFRead(da, localSolution, &solutionArray);
	CHKERRQ(ierr);

	// Store the concentration and other values over the grid
	double xeConcentration = 0.0, bubbleConcentration = 0.0, radii = 0.0,
		   partialBubbleConcentration = 0.0, partialRadii = 0.0;

	// Get the minimum size for the radius
	auto minSizes = solverHandler.getMinSizes();

	// Get Xe_1
	Composition xeComp = Composition::zero();
	xeComp[Spec::Xe] = 1;
	auto xeCluster = network.findCluster(xeComp, plsm::onHost);
	auto xeId = xeCluster.getId();

	// Loop on the grid
	for (PetscInt yj = ys; yj < ys + ym; yj++)
		for (PetscInt xi = xs; xi < xs + xm; xi++) {
			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[yj][xi];

			using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
				Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(gridPointSolution, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);

			double hx = grid[xi + 1] - grid[xi];

			// Get the concentrations
			xeConcentration +=
				network.getTotalAtomConcentration(dConcs, Spec::Xe, 1) * hx *
				hy;
			bubbleConcentration +=
				network.getTotalConcentration(dConcs, Spec::Xe, 1) * hx * hy;
			radii += network.getTotalRadiusConcentration(dConcs, Spec::Xe, 1) *
				hx * hy;
			partialBubbleConcentration +=
				network.getTotalConcentration(dConcs, Spec::Xe, minSizes[0]) *
				hx * hy;
			partialRadii += network.getTotalRadiusConcentration(
								dConcs, Spec::Xe, minSizes[0]) *
				hx * hy;

			// Set the volume fraction
			double volumeFrac =
				network.getTotalVolumeFraction(dConcs, Spec::Xe, minSizes[0]);
			solverHandler.setVolumeFraction(volumeFrac, xi - xs, yj - ys);
			// Set the monomer concentration
			solverHandler.setMonomerConc(
				gridPointSolution[xeCluster.getId()], xi - xs, yj - ys);
		}

	// Get the current process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Sum all the concentrations through MPI reduce
	std::array<double, 5> myConcData{xeConcentration, bubbleConcentration,
		radii, partialBubbleConcentration, partialRadii};
	std::array<double, 5> totalConcData{0.0, 0.0, 0.0, 0.0, 0.0};
	MPI_Reduce(myConcData.data(), totalConcData.data(), myConcData.size(),
		MPI_DOUBLE, MPI_SUM, 0, xolotlComm);

	// GB
	// Get the delta time from the previous timestep to this timestep
	double dt = time - solverHandler.getPreviousTime();
	// Sum and gather the previous flux
	double globalXeFlux = 0.0;
	// Get the vector from the solver handler
	auto gbVector = solverHandler.getGBVector();
	// Get the previous Xe flux vector
	auto& localNE = solverHandler.getLocalNE();
	// Loop on the GB
	for (auto const& pair : gbVector) {
		// Middle
		int xi = std::get<0>(pair);
		int yj = std::get<1>(pair);
		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
			double previousXeFlux = std::get<1>(localNE[xi - xs][yj - ys][0]);
			globalXeFlux += previousXeFlux * (grid[xi + 1] - grid[xi]) * hy;
			// Set the amount in the vector we keep
			solverHandler.setLocalXeRate(previousXeFlux * dt, xi - xs, yj - ys);
		}
	}
	double totalXeFlux = 0.0;
	MPI_Reduce(
		&globalXeFlux, &totalXeFlux, 1, MPI_DOUBLE, MPI_SUM, 0, xolotlComm);
	// Master process
	if (procId == 0) {
		// Get the previous value of Xe that went to the GB
		double nXenon = solverHandler.getNXeGB();
		// Compute the total number of Xe that went to the GB
		nXenon += totalXeFlux * dt;
		solverHandler.setNXeGB(nXenon);
	}

	// Loop on the GB
	for (auto const& pair : gbVector) {
		// Local rate
		double localRate = 0.0;
		// Define left and right with reference to the middle point
		// Middle
		int xi = std::get<0>(pair);
		int yj = std::get<1>(pair);
		double hxLeft = grid[xi + 1] - grid[xi];
		double hxRight = grid[xi + 2] - grid[xi + 1];
		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
			// X segment
			// Left
			xi = std::get<0>(pair) - 1;
			// Compute the flux coming from the left
			localRate += solutionArray[yj][xi][xeId] *
				xeCluster.getDiffusionCoefficient(xi + 1 - xs) * 2.0 /
				((hxLeft + hxRight) * hxLeft);

			// Right
			xi = std::get<0>(pair) + 1;
			// Compute the flux coming from the right
			localRate += solutionArray[yj][xi][xeId] *
				xeCluster.getDiffusionCoefficient(xi + 1 - xs) * 2.0 /
				((hxLeft + hxRight) * hxRight);

			// Y segment
			// Bottom
			xi = std::get<0>(pair);
			yj = std::get<1>(pair) - 1;
			// Compute the flux coming from the bottom
			localRate += solutionArray[yj][xi][xeId] *
				xeCluster.getDiffusionCoefficient(xi + 1 - xs) / (hy * hy);

			// Top
			yj = std::get<1>(pair) + 1;
			// Compute the flux coming from the top
			localRate += solutionArray[yj][xi][xeId] *
				xeCluster.getDiffusionCoefficient(xi + 1 - xs) / (hy * hy);

			// Middle
			xi = std::get<0>(pair);
			yj = std::get<1>(pair);
			solverHandler.setPreviousXeFlux(localRate, xi - xs, yj - ys);
		}
	}

	// Master process
	if (procId == 0) {
		// Compute the total surface irradiated
		double surface = (double)My * hy;
		// Get the number of Xe that went to the GB
		double nXenon = solverHandler.getNXeGB();

		totalConcData[0] = totalConcData[0] / surface;

		// Print the result
		std::cout << "\nTime: " << time << std::endl;
		std::cout << "Xenon concentration = " << totalConcData[0] << std::endl;
		std::cout << "Xenon GB = " << nXenon / surface << std::endl
				  << std::endl;

		// Make sure the average partial radius makes sense
		double averagePartialRadius = 0.0;
		if (totalConcData[3] > 1.e-16) {
			averagePartialRadius = totalConcData[4] / totalConcData[3];
		}

		// Uncomment to write the retention and the fluence in a file
		std::ofstream outputFile;
		outputFile.open("retentionOut.txt", std::ios::app);
		outputFile << time << " " << totalConcData[0] << " "
				   << totalConcData[2] / totalConcData[1] << " "
				   << averagePartialRadius << " " << nXenon / surface
				   << std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, localSolution, &solutionArray);
	CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &localSolution);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorSurface2D")
/**
 * This is a monitoring method that will save 2D plots of the concentration of
 * a specific cluster at each grid point.
 */
PetscErrorCode
monitorSurface2D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	// Initial declarations
	PetscErrorCode ierr;
	const double ***solutionArray, *gridPointSolution;
	PetscInt xs, xm, Mx, ys, ym, My;
	double x = 0.0, y = 0.0;

	PetscFunctionBeginUser;

	// Don't do anything if it is not on the stride
	if (timestep % 10 != 0)
		PetscFunctionReturn(0);

	// Gets the process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);
	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the physical grid in the x direction
	auto grid = solverHandler.getXGrid();

	// Setup step size variables
	double hy = solverHandler.getStepSizeY();

	// Choice of the cluster to be plotted
	int iCluster = 0;

	// Create a DataPoint vector to store the data to give to the data provider
	// for the visualization
	auto myPoints =
		std::make_shared<std::vector<viz::dataprovider::DataPoint>>();
	// Create a point here so that it is not created and deleted in the loop
	viz::dataprovider::DataPoint thePoint;

	// Loop on the full grid
	for (PetscInt j = 0; j < My; j++) {
		for (PetscInt i = 0; i < Mx; i++) {
			// If it is the locally owned part of the grid
			if (i >= xs && i < xs + xm && j >= ys && j < ys + ym) {
				// Get the pointer to the beginning of the solution data for
				// this grid point
				gridPointSolution = solutionArray[j][i];
				// Compute x and y
				x = (grid[i] + grid[i + 1]) / 2.0 - grid[1];
				y = (double)j * hy;

				// If it is procId 0 just store the value in the myPoints vector
				if (procId == 0) {
					// Modify the Point with the gridPointSolution[iCluster] as
					// the value and add it to myPoints
					thePoint.value = gridPointSolution[iCluster];
					thePoint.t = time;
					thePoint.x = x;
					thePoint.y = y;
					myPoints->push_back(thePoint);
				}
				// Else, the values must be sent to procId 0
				else {
					// Send the value of the local position to the master
					// process
					MPI_Send(&x, 1, MPI_DOUBLE, 0, 10, xolotlComm);
					// Send the value of the local position to the master
					// process
					MPI_Send(&y, 1, MPI_DOUBLE, 0, 11, xolotlComm);

					// Send the value of the concentration to the master process
					MPI_Send(&gridPointSolution[iCluster], 1, MPI_DOUBLE, 0, 12,
						xolotlComm);
				}
			}
			// Else if it is NOT the locally owned part of the grid but still
			// procId == 0, it should receive the values for the point and add
			// them to myPoint
			else if (procId == 0) {
				// Get the position
				MPI_Recv(&x, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 10, xolotlComm,
					MPI_STATUS_IGNORE);
				MPI_Recv(&y, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 11, xolotlComm,
					MPI_STATUS_IGNORE);

				// and the concentration
				double conc = 0.0;
				MPI_Recv(&conc, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 12, xolotlComm,
					MPI_STATUS_IGNORE);

				// Modify the Point with the received values and add it to
				// myPoints
				thePoint.value = conc;
				thePoint.t = time;
				thePoint.x = x;
				thePoint.y = y;
				myPoints->push_back(thePoint);
			}

			// Wait for everybody at each grid point
			MPI_Barrier(xolotlComm);
		}
	}

	// Plot everything from procId == 0
	if (procId == 0) {
		// Get the data provider and give it the points
		surfacePlot2D->getDataProvider()->setDataPoints(myPoints);

		// Change the title of the plot and the name of the data
		std::stringstream title;
		// TODO: get the name or comp of the cluster
		title << "First Cluster";
		surfacePlot2D->getDataProvider()->setDataName(title.str());
		title << " concentration";
		surfacePlot2D->plotLabelProvider->titleLabel = title.str();
		// Give the time to the label provider
		std::stringstream timeLabel;
		timeLabel << "time: " << std::setprecision(4) << time << "s";
		surfacePlot2D->plotLabelProvider->timeLabel = timeLabel.str();
		// Get the current time step
		PetscReal currentTimeStep;
		ierr = TSGetTimeStep(ts, &currentTimeStep);
		CHKERRQ(ierr);
		// Give the timestep to the label provider
		std::stringstream timeStepLabel;
		timeStepLabel << "dt: " << std::setprecision(4) << currentTimeStep
					  << "s";
		surfacePlot2D->plotLabelProvider->timeStepLabel = timeStepLabel.str();

		// Render and save in file
		std::stringstream fileName;
		fileName << "surface_TS" << timestep << ".png";
		surfacePlot2D->write(fileName.str());
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "eventFunction2D")
/**
 * This is a method that checks if the surface should move or bursting happen
 */
PetscErrorCode
eventFunction2D(TS ts, PetscReal time, Vec solution, PetscScalar* fvalue, void*)
{
	// Initial declaration
	PetscErrorCode ierr;
	double ***solutionArray, *gridPointSolution;
	PetscInt xs, xm, xi, Mx, ys, ym, yj, My;
	fvalue[0] = 1.0, fvalue[1] = 1.0;
	depthPositions2D.clear();

	PetscFunctionBeginUser;

	// Gets the process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);

	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the network
	auto& network = solverHandler.getNetwork();

	// Get the physical grid
	auto grid = solverHandler.getXGrid();
	// Get the step size in Y
	double hy = solverHandler.getStepSizeY();

	// Get the flux handler to know the flux amplitude.
	auto fluxHandler = solverHandler.getFluxHandler();
	double heliumFluxAmplitude = fluxHandler->getFluxAmplitude();

	// Get the delta time from the previous timestep to this timestep
	double dt = time - solverHandler.getPreviousTime();

	// Work of the moving surface first
	if (solverHandler.moveSurface()) {
		// Write the initial surface positions
		if (procId == 0 && util::equal(time, 0.0)) {
			std::ofstream outputFile;
			outputFile.open("surface.txt", std::ios::app);
			outputFile << time << " ";

			// Loop on the possible yj
			for (yj = 0; yj < My; yj++) {
				// Get the position of the surface at yj
				int surfacePos = solverHandler.getSurfacePosition(yj);
				outputFile << grid[surfacePos + 1] - grid[1] << " ";
			}
			outputFile << std::endl;
			outputFile.close();
		}

		// Get the initial vacancy concentration
		double initialVConc = solverHandler.getInitialVConc();

		// Loop on the possible yj
		for (yj = 0; yj < My; yj++) {
			// Compute the total density of intersitials that escaped from the
			// surface since last timestep using the stored flux
			nInterstitial2D[yj] += previousIFlux2D[yj] * dt;

			// Remove the sputtering yield since last timestep
			nInterstitial2D[yj] -=
				sputteringYield2D * heliumFluxAmplitude * dt * hy;

			// Get the position of the surface at yj
			const int surfacePos = solverHandler.getSurfacePosition(yj);
			xi = surfacePos + solverHandler.getLeftOffset();
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi - 1 >= 0 && xi < Mx) {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else if (xi - 1 < 0) {
				hxLeft = (grid[xi + 1] + grid[xi]) / 2.0;
				hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
			}
			else {
				hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
				hxRight = (grid[xi + 1] - grid[xi]) / 2;
			}

			// Initialize the value for the flux
			double newFlux = 0.0;

			// if xi is on this process
			if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
				// Get the concentrations at xi = surfacePos + 1
				gridPointSolution = solutionArray[yj][xi];

				// Factor for finite difference
				double factor = 2.0 / (hxLeft + hxRight);

				// Consider each interstitial cluster.
				for (int i = 0; i < iClusterIds2D.size(); i++) {
					auto currId = iClusterIds2D[i];
					// Get the cluster
					auto cluster = network.getClusterCommon(currId);
					// Get its concentration
					double conc = gridPointSolution[currId];
					// Get its size and diffusion coefficient
					int size = i + 1;
					double coef = cluster.getDiffusionCoefficient(xi - xs);
					// Compute the flux going to the left
					newFlux += (double)size * factor * coef * conc * hxLeft;
				}
			}

			// Check if the surface on the left and/or right sides are higher
			// than at this grid point
			int yLeft = yj - 1, yRight = yj + 1;
			if (yLeft < 0)
				yLeft = My - 1; // Periodicity
			if (yRight == My)
				yRight = 0; // Periodicity
			// We want the position just at the surface now
			xi = surfacePos;
			// Do the left side first
			if (solverHandler.getSurfacePosition(yLeft) > surfacePos) {
				// if we are on the right process
				if (xi >= xs && xi < xs + xm && yLeft >= ys &&
					yLeft < ys + ym) {
					// Get the concentrations at xi = surfacePos
					gridPointSolution = solutionArray[yLeft][xi];

					// Loop on all the interstitial clusters to add the
					// contribution from the left side
					for (int i = 0; i < iClusterIds2D.size(); i++) {
						auto currId = iClusterIds2D[i];
						// Get the cluster
						auto cluster = network.getClusterCommon(currId);
						// Get its concentration
						double conc = gridPointSolution[currId];
						// Get its size and diffusion coefficient
						int size = i + 1;
						double coef = cluster.getDiffusionCoefficient(xi - xs);
						// Compute the flux
						newFlux += ((double)size * coef * conc * hxLeft) / hy;
					}
				}
			}
			// Now do the right side
			if (solverHandler.getSurfacePosition(yRight) > surfacePos) {
				// if we are on the right process
				if (xi >= xs && xi < xs + xm && yRight >= ys &&
					yRight < ys + ym) {
					// Get the concentrations at xi = surfacePos + 1
					gridPointSolution = solutionArray[yRight][xi];

					// Loop on all the interstitial clusters to add the
					// contribution from the right side
					for (int i = 0; i < iClusterIds2D.size(); i++) {
						auto currId = iClusterIds2D[i];
						// Get the cluster
						auto cluster = network.getClusterCommon(currId);
						// Get its concentration
						double conc = gridPointSolution[currId];
						// Get its size and diffusion coefficient
						int size = i + 1;
						double coef = cluster.getDiffusionCoefficient(xi - xs);
						// Compute the flux
						newFlux += ((double)size * coef * conc * hxLeft) / hy;
					}
				}
			}

			// Gather newFlux values at this position
			double newTotalFlux = 0.0;
			MPI_Allreduce(
				&newFlux, &newTotalFlux, 1, MPI_DOUBLE, MPI_SUM, xolotlComm);

			// Update the previous flux
			previousIFlux2D[yj] = newTotalFlux;

			// Compare nInterstitials to the threshold to know if we should move
			// the surface

			// The density of tungsten is 62.8 atoms/nm3, thus the threshold is
			double threshold = (62.8 - initialVConc) * hxLeft * hy;
			if (nInterstitial2D[yj] > threshold) {
				// The surface is moving
				fvalue[0] = 0.0;
			}

			// Moving the surface back
			else if (nInterstitial2D[yj] < -threshold / 10.0) {
				// The surface is moving
				fvalue[0] = 0.0;
			}
		}
	}

	// Now work on the bubble bursting
	if (solverHandler.burstBubbles()) {
		using NetworkType = core::network::PSIReactionNetwork<
			core::network::PSIFullSpeciesList>;
		using Spec = typename NetworkType::Species;
		auto psiNetwork = dynamic_cast<NetworkType*>(&network);
		auto dof = network.getDOF();

		// Compute the prefactor for the probability (arbitrary)
		double prefactor = heliumFluxAmplitude * dt * 0.1;

		// The depth parameter to know where the bursting should happen
		double depthParam = solverHandler.getTauBursting(); // nm

		// For now we are not bursting
		bool burst = false;

		// Loop on the full grid
		for (yj = 0; yj < My; yj++) {
			// Get the surface position
			int surfacePos = solverHandler.getSurfacePosition(yj);
			for (xi = surfacePos + solverHandler.getLeftOffset();
				 xi < Mx - solverHandler.getRightOffset(); xi++) {
				// If this is the locally owned part of the grid
				if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
					// Get the pointer to the beginning of the solution data for
					// this grid point
					gridPointSolution = solutionArray[yj][xi];

					using HostUnmanaged = Kokkos::View<double*,
						Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
					auto hConcs = HostUnmanaged(gridPointSolution, dof);
					auto dConcs = Kokkos::View<double*>("Concentrations", dof);
					deep_copy(dConcs, hConcs);

					// Get the distance from the surface
					double distance =
						(grid[xi] + grid[xi + 1]) / 2.0 - grid[surfacePos + 1];

					// Compute the helium density at this grid point
					double heDensity = psiNetwork->getTotalAtomConcentration(
						dConcs, Spec::He, 1);

					// Compute the radius of the bubble from the number of
					// helium
					double nV = heDensity * (grid[xi + 1] - grid[xi]) / 4.0;
					//				double nV = pow(heDensity / 5.0, 1.163) *
					//(grid[xi
					//+ 1] - grid[xi]);

					double latticeParam = network.getLatticeParameter();
					double tlcCubed =
						latticeParam * latticeParam * latticeParam;
					double radius = (sqrt(3.0) / 4) * latticeParam +
						cbrt((3.0 * tlcCubed * nV) / (8.0 * core::pi)) -
						cbrt((3.0 * tlcCubed) / (8.0 * core::pi));

					// If the radius is larger than the distance to the surface,
					// burst
					if (radius > distance) {
						burst = true;
						depthPositions2D.push_back(std::make_pair(yj, xi));
						// Exit the loop
						continue;
					}
					// Add randomness
					double prob = prefactor *
						(1.0 - (distance - radius) / distance) *
						std::min(1.0,
							exp(-(distance - depthParam) / (depthParam * 2.0)));
					double test = solverHandler.getRNG().GetRandomDouble();

					if (prob > test) {
						burst = true;
						depthPositions2D.push_back(std::make_pair(yj, xi));
					}
				}
			}
		}

		// If at least one grid point is bursting
		if (burst) {
			// The event is happening
			fvalue[1] = 0.0;
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "postEventFunction2D")
/**
 * This is a method that moves the surface or burst bubbles
 */
PetscErrorCode
postEventFunction2D(TS ts, PetscInt nevents, PetscInt eventList[],
	PetscReal time, Vec solution, PetscBool, void*)
{
	// Initial declaration
	PetscErrorCode ierr;
	double ***solutionArray, *gridPointSolution;
	PetscInt xs, xm, xi, Mx, ys, ym, yj, My;

	PetscFunctionBeginUser;

	// Call monitor time hear because it is skipped when post event is used
	ierr = computeFluence(ts, 0, time, solution, NULL);
	CHKERRQ(ierr);
	ierr = monitorTime(ts, 0, time, solution, NULL);
	CHKERRQ(ierr);

	// Check if the surface has moved
	if (nevents == 0)
		PetscFunctionReturn(0);

	// Gets the process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
	CHKERRQ(ierr);

	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the network
	auto& network = solverHandler.getNetwork();
	int dof = network.getDOF();

	// Get the physical grid
	auto grid = solverHandler.getXGrid();
	// Get the step size in Y
	double hy = solverHandler.getStepSizeY();

	// Take care of bursting
	using NetworkType =
		core::network::PSIReactionNetwork<core::network::PSIFullSpeciesList>;
	using Spec = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;
	auto psiNetwork = dynamic_cast<NetworkType*>(&network);

	// Loop on each bursting depth
	for (int i = 0; i < depthPositions2D.size(); i++) {
		// Get the coordinates of the point
		int xi = depthPositions2D[i].second, yj = depthPositions2D[i].first;
		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[yj][xi];

		// Get the surface position
		int surfacePos = solverHandler.getSurfacePosition(yj);
		// Get the distance from the surface
		double distance =
			(grid[xi] - grid[xi + 1]) / 2.0 - grid[surfacePos + 1];

		std::cout << "bursting at: " << yj * hy << " " << distance << std::endl;

		// Pinhole case
		// Loop on every cluster
		for (unsigned int i = 0; i < network.getNumClusters(); i++) {
			const auto& clReg =
				psiNetwork->getCluster(i, plsm::onHost).getRegion();
			// Non-grouped clusters
			if (clReg.isSimplex()) {
				// Get the composition
				Composition comp = clReg.getOrigin();
				// Pure He, D, or T case
				if (comp.isOnAxis(Spec::He) || comp.isOnAxis(Spec::D) ||
					comp.isOnAxis(Spec::T)) {
					// Reset concentration
					gridPointSolution[i] = 0.0;
				}
				// Mixed cluster case
				else if (!comp.isOnAxis(Spec::V) && !comp.isOnAxis(Spec::I)) {
					// Transfer concentration to V of the same size
					Composition vComp = Composition::zero();
					vComp[Spec::V] = comp[Spec::V];
					auto vCluster =
						psiNetwork->findCluster(vComp, plsm::onHost);
					gridPointSolution[vCluster.getId()] += gridPointSolution[i];
					gridPointSolution[i] = 0.0;
				}
			}
			// Grouped clusters
			else {
				// Get the factor
				double concFactor = clReg.volume() / clReg[Spec::V].length();
				// Loop on the Vs
				for (auto j : makeIntervalRange(clReg[Spec::V])) {
					// Transfer concentration to V of the same size
					Composition vComp = Composition::zero();
					vComp[Spec::V] = j;
					auto vCluster =
						psiNetwork->findCluster(vComp, plsm::onHost);
					// TODO: refine formula with V moment
					gridPointSolution[vCluster.getId()] +=
						gridPointSolution[i] * concFactor;
				}

				// Reset the concentration and moments
				gridPointSolution[i] = 0.0;
				auto momentIds =
					psiNetwork->getCluster(i, plsm::onHost).getMomentIds();
				for (std::size_t j = 0; j < momentIds.extent(0); j++) {
					gridPointSolution[momentIds(j)] = 0.0;
				}
			}
		}
	}

	// Now takes care of moving surface
	bool moving = false;
	for (int i = 0; i < nevents; i++) {
		if (eventList[i] == 0)
			moving = true;
	}

	// Skip if nothing is moving
	if (!moving) {
		// Restore the solutionArray
		ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
		CHKERRQ(ierr);

		PetscFunctionReturn(0);
	}

	// Get the initial vacancy concentration
	double initialVConc = solverHandler.getInitialVConc();

	// Loop on the possible yj
	for (yj = 0; yj < My; yj++) {
		// Get the position of the surface at yj
		int surfacePos = solverHandler.getSurfacePosition(yj);
		xi = surfacePos + solverHandler.getLeftOffset();

		// The density of tungsten is 62.8 atoms/nm3, thus the threshold is
		double threshold =
			(62.8 - initialVConc) * (grid[xi] - grid[xi - 1]) * hy;

		// Move the surface up
		if (nInterstitial2D[yj] > threshold) {
			int nGridPoints = 0;
			// Move the surface up until it is smaller than the next threshold
			while (nInterstitial2D[yj] > threshold) {
				// Move the surface higher
				surfacePos--;
				xi = surfacePos + solverHandler.getLeftOffset();
				nGridPoints++;
				// Update the number of interstitials
				nInterstitial2D[yj] -= threshold;
				// Update the thresold
				threshold =
					(62.8 - initialVConc) * (grid[xi] - grid[xi - 1]) * hy;
			}

			// Throw an exception if the position is negative
			if (surfacePos < 0) {
				PetscBool flagCheck;
				ierr = PetscOptionsHasName(
					NULL, NULL, "-check_collapse", &flagCheck);
				CHKERRQ(ierr);
				if (flagCheck) {
					// Write the convergence reason
					std::ofstream outputFile;
					outputFile.open("solverStatus.txt");
					outputFile << "overgrid" << std::endl;
					outputFile.close();
				}
				throw std::string("\nxolotlSolver::Monitor2D: The surface is "
								  "trying to go outside of the grid!!");
			}

			// Printing information about the extension of the material
			if (procId == 0) {
				std::cout << "Adding " << nGridPoints
						  << " points to the grid on " << yj * hy
						  << " at time: " << time << " s." << std::endl;
			}

			// Set it in the solver
			solverHandler.setSurfacePosition(surfacePos, yj);

			// Initialize the vacancy concentration and the temperature on the
			// new grid points Get the single vacancy ID
			auto singleVacancyCluster = network.getSingleVacancy();
			auto vacancyIndex = core::network::IReactionNetwork::invalidIndex();
			if (singleVacancyCluster.getId() !=
				core::network::IReactionNetwork::invalidIndex())
				vacancyIndex = singleVacancyCluster.getId();
			// Get the surface temperature
			double temp = 0.0;
			if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
				temp = solutionArray[yj][xi][dof];
			}
			double surfTemp = 0.0;
			MPI_Allreduce(&temp, &surfTemp, 1, MPI_DOUBLE, MPI_SUM, xolotlComm);
			// Loop on the new grid points
			while (nGridPoints >= 0) {
				// Position of the newly created grid point
				xi = surfacePos + nGridPoints;

				// If xi is on this process
				if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
					// Get the concentrations
					gridPointSolution = solutionArray[yj][xi];

					// Set the new surface temperature
					gridPointSolution[dof] = surfTemp;

					if (vacancyIndex > 0 && nGridPoints > 0) {
						// Initialize the vacancy concentration
						gridPointSolution[vacancyIndex] = initialVConc;
					}
				}

				// Decrease the number of grid points
				--nGridPoints;
			}
		}

		// Moving the surface back
		else if (nInterstitial2D[yj] < -threshold / 10.0) {
			// Move it back as long as the number of interstitials in negative
			while (nInterstitial2D[yj] < 0.0) {
				// Compute the threshold to a deeper grid point
				threshold =
					(62.8 - initialVConc) * (grid[xi + 1] - grid[xi]) * hy;
				// Set all the concentrations to 0.0 at xi = surfacePos + 1
				// if xi is on this process
				if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym) {
					// Get the concentrations at xi = surfacePos + 1
					gridPointSolution = solutionArray[yj][xi];
					// Loop on DOF
					for (int i = 0; i < dof; i++) {
						gridPointSolution[i] = 0.0;
					}
				}

				// Move the surface deeper
				surfacePos++;
				xi = surfacePos + solverHandler.getLeftOffset();
				// Update the number of interstitials
				nInterstitial2D[yj] += threshold;
			}

			// Printing information about the extension of the material
			if (procId == 0) {
				std::cout << "Removing grid points to the grid on " << yj * hy
						  << " at time: " << time << " s." << std::endl;
			}

			// Set it in the solver
			solverHandler.setSurfacePosition(surfacePos, yj);
		}
	}

	// Get the modified trap-mutation handler to reinitialize it
	auto mutationHandler = solverHandler.getMutationHandler();
	auto advecHandlers = solverHandler.getAdvectionHandlers();

	// Get the vector of positions of the surface
	std::vector<int> surfaceIndices;
	for (PetscInt i = 0; i < My; i++) {
		surfaceIndices.push_back(solverHandler.getSurfacePosition(i));
	}

	mutationHandler->initializeIndex2D(surfaceIndices,
		solverHandler.getNetwork(), advecHandlers, grid, xm, xs, ym, hy, ys);

	// Write the surface positions
	if (procId == 0) {
		std::ofstream outputFile;
		outputFile.open("surface.txt", std::ios::app);
		outputFile << time << " ";

		// Loop on the possible yj
		for (yj = 0; yj < My; yj++) {
			// Get the position of the surface at yj
			int surfacePos = solverHandler.getSurfacePosition(yj);
			outputFile << grid[surfacePos + 1] - grid[1] << " ";
		}
		outputFile << std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 * This operation sets up different monitors
 *  depending on the options.
 * @param ts The time stepper
 * @return A standard PETSc error code
 */
PetscErrorCode
setupPetsc2DMonitor(TS ts)
{
	PetscErrorCode ierr;

	auto handlerRegistry = perf::getHandlerRegistry();
	gbTimer = handlerRegistry->getTimer("monitor2D:GB");

	// Get the process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the xolotlViz handler registry
	auto vizHandlerRegistry = viz::VizHandlerRegistry::get();

	// Flags to launch the monitors or not
	PetscBool flagCheck, flagPerf, flagHeRetention, flagXeRetention, flagStatus,
		flag2DPlot, flagLargest;

	// Check the option -check_collapse
	ierr = PetscOptionsHasName(NULL, NULL, "-check_collapse", &flagCheck);
	checkPetscError(ierr,
		"setupPetsc2DMonitor: PetscOptionsHasName (-check_collapse) failed.");

	// Check the option -plot_perf
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_perf", &flagPerf);
	checkPetscError(
		ierr, "setupPetsc2DMonitor: PetscOptionsHasName (-plot_perf) failed.");

	// Check the option -plot_2d
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_2d", &flag2DPlot);
	checkPetscError(
		ierr, "setupPetsc2DMonitor: PetscOptionsHasName (-plot_2d) failed.");

	// Check the option -helium_retention
	ierr =
		PetscOptionsHasName(NULL, NULL, "-helium_retention", &flagHeRetention);
	checkPetscError(ierr,
		"setupPetsc2DMonitor: PetscOptionsHasName (-helium_retention) failed.");

	// Check the option -xenon_retention
	ierr =
		PetscOptionsHasName(NULL, NULL, "-xenon_retention", &flagXeRetention);
	checkPetscError(ierr,
		"setupPetsc2DMonitor: PetscOptionsHasName (-xenon_retention) failed.");

	// Check the option -start_stop
	ierr = PetscOptionsHasName(NULL, NULL, "-start_stop", &flagStatus);
	checkPetscError(
		ierr, "setupPetsc2DMonitor: PetscOptionsHasName (-start_stop) failed.");

	// Check the option -largest_conc
	ierr = PetscOptionsHasName(NULL, NULL, "-largest_conc", &flagLargest);
	checkPetscError(ierr,
		"setupPetsc2DMonitor: PetscOptionsHasName (-largest_conc) failed.");

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the network and its size
	auto& network = solverHandler.getNetwork();

	// Determine if we have an existing restart file,
	// and if so, it it has had timesteps written to it.
	std::unique_ptr<io::XFile> networkFile;
	std::unique_ptr<io::XFile::TimestepGroup> lastTsGroup;
	std::string networkName = solverHandler.getNetworkName();
	bool hasConcentrations = false;
	if (not networkName.empty()) {
		networkFile.reset(new io::XFile(networkName));
		auto concGroup = networkFile->getGroup<io::XFile::ConcentrationGroup>();
		hasConcentrations = (concGroup and concGroup->hasTimesteps());
		if (hasConcentrations) {
			lastTsGroup = concGroup->getLastTimestepGroup();
		}
	}

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);
	checkPetscError(ierr, "setupPetsc2DMonitor: TSGetDM failed.");

	// Get the total size of the grid
	PetscInt Mx, My;
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);
	checkPetscError(ierr, "setupPetsc2DMonitor: DMDAGetInfo failed.");

	// Set the post step processing to stop the solver if the time step
	// collapses
	if (flagCheck) {
		// Find the threshold
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-check_collapse", &timeStepThreshold, &flag);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: PetscOptionsGetReal (-check_collapse) "
			"failed.");
		if (!flag)
			timeStepThreshold = 1.0e-16;

		// Set the post step process that tells the solver when to stop if the
		// time step collapse
		ierr = TSSetPostStep(ts, checkTimeStep);
		checkPetscError(
			ierr, "setupPetsc2DMonitor: TSSetPostStep (checkTimeStep) failed.");
	}

	// Set the monitor to save the status of the simulation in hdf5 file
	if (flagStatus) {
		// Find the stride to know how often the HDF5 file has to be written
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-start_stop", &hdf5Stride2D, &flag);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: PetscOptionsGetReal (-start_stop) failed.");
		if (!flag)
			hdf5Stride2D = 1.0;

		if (hasConcentrations) {
			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			hdf5Previous2D = (int)(previousTime / hdf5Stride2D);
		}

		// Don't do anything if both files have the same name
		if (hdf5OutputName2D != solverHandler.getNetworkName()) {
			// Get the solver handler
			auto& solverHandler = PetscSolver::getSolverHandler();

			// Get the physical grid in the x direction
			auto grid = solverHandler.getXGrid();

			// Setup step size variables
			double hy = solverHandler.getStepSizeY();

			// Create and initialize a checkpoint file.
			// We do this in its own scope so that the file
			// is closed when the file object goes out of scope.
			// We want it to close before we (potentially) copy
			// the network from another file using a single-process
			// MPI communicator.
			{
				io::XFile checkpointFile(
					hdf5OutputName2D, grid, xolotlComm, My, hy);
			}

			// Copy the network group from the given file (if it has one).
			// We open the files using a single-process MPI communicator
			// because it is faster for a single process to do the
			// copy with HDF5's H5Ocopy implementation than it is
			// when all processes call the copy function.
			// The checkpoint file must be closed before doing this.
			writeNetwork(xolotlComm, solverHandler.getNetworkName(),
				hdf5OutputName2D, network);
		}

		// startStop2D will be called at each timestep
		ierr = TSMonitorSet(ts, startStop2D, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc2DMonitor: TSMonitorSet (startStop2D) failed.");
	}

	// If the user wants the surface to be able to move or bursting
	if (solverHandler.moveSurface() || solverHandler.burstBubbles()) {
		// Surface
		if (solverHandler.moveSurface()) {
			// Initialize nInterstitial2D and previousIFlux2D before monitoring
			// the interstitial flux
			for (PetscInt j = 0; j < My; j++) {
				nInterstitial2D.push_back(0.0);
				previousIFlux2D.push_back(0.0);
			}

			// Get the interstitial information at the surface if concentrations
			// were stored
			if (hasConcentrations) {
				assert(lastTsGroup);

				// Get the interstitial quantity from the HDF5 file
				nInterstitial2D = lastTsGroup->readData2D("nInterstitial");
				// Get the previous I flux from the HDF5 file
				previousIFlux2D = lastTsGroup->readData2D("previousIFlux");
				// Get the previous time from the HDF5 file
				double previousTime = lastTsGroup->readPreviousTime();
				solverHandler.setPreviousTime(previousTime);
			}

			// Get the sputtering yield
			sputteringYield2D = solverHandler.getSputteringYield();

			// Master process
			if (procId == 0) {
				// Clear the file where the surface will be written
				std::ofstream outputFile;
				outputFile.open("surface.txt");
				outputFile.close();
			}
		}

		// Bursting
		if (solverHandler.burstBubbles()) {
			// No need to seed the random number generator here.
			// The solver handler has already done it.
		}

		// Set directions and terminate flags for the surface event
		PetscInt direction[2];
		PetscBool terminate[2];
		direction[0] = 0, direction[1] = 0;
		terminate[0] = PETSC_FALSE, terminate[1] = PETSC_FALSE;
		// Set the TSEvent
		ierr = TSSetEventHandler(ts, 2, direction, terminate, eventFunction2D,
			postEventFunction2D, NULL);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: TSSetEventHandler (eventFunction2D) failed.");
	}

	// Set the monitor to save performance plots (has to be in parallel)
	if (flagPerf) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a ScatterPlot
			perfPlot =
				vizHandlerRegistry->getPlot("perfPlot", viz::PlotType::SCATTER);

			// Create and set the label provider
			auto labelProvider =
				std::make_shared<viz::LabelProvider>("labelProvider");
			labelProvider->axis1Label = "Process ID";
			labelProvider->axis2Label = "Solver Time";

			// Give it to the plot
			perfPlot->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider =
				std::make_shared<viz::dataprovider::CvsXDataProvider>(
					"dataProvider");

			// Give it to the plot
			perfPlot->setDataProvider(dataProvider);
		}

		// monitorPerf will be called at each timestep
		ierr = TSMonitorSet(ts, monitorPerf, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc2DMonitor: TSMonitorSet (monitorPerf) failed.");
	}

	// Set the monitor to compute the helium fluence for the retention
	// calculation
	if (flagHeRetention) {
		// Check if we have a free surface at the bottom
		if (solverHandler.getRightOffset() == 1) {
			// Initialize n2D and previousFlux2D before monitoring the fluxes
			for (PetscInt j = 0; j < My; j++) {
				nHelium2D.push_back(0.0);
				previousHeFlux2D.push_back(0.0);
				nDeuterium2D.push_back(0.0);
				previousDFlux2D.push_back(0.0);
				nTritium2D.push_back(0.0);
				previousTFlux2D.push_back(0.0);
			}
		}

		// Get the previous time if concentrations were stored and initialize
		// the fluence
		if (hasConcentrations) {
			assert(lastTsGroup);

			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			// Initialize the fluence
			auto fluxHandler = solverHandler.getFluxHandler();
			// Increment the fluence with the value at this current timestep
			fluxHandler->computeFluence(previousTime);

			// If the bottom is a free surface
			if (solverHandler.getRightOffset() == 1) {
				// Read about the impurity fluxes in the bulk
				nHelium2D = lastTsGroup->readData2D("nHelium");
				previousHeFlux2D = lastTsGroup->readData2D("previousHeFlux");
				nDeuterium2D = lastTsGroup->readData2D("nDeuterium");
				previousDFlux2D = lastTsGroup->readData2D("previousDFlux");
				nTritium2D = lastTsGroup->readData2D("nTritium");
				previousTFlux2D = lastTsGroup->readData2D("previousTFlux");
			}
		}

		// computeFluence will be called at each timestep
		ierr = TSMonitorSet(ts, computeFluence, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc2DMonitor: TSMonitorSet (computeFluence) failed.");

		// computeHeliumRetention2D will be called at each timestep
		ierr = TSMonitorSet(ts, computeHeliumRetention2D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: TSMonitorSet (computeHeliumRetention2D) "
			"failed.");

		// Master process
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("retentionOut.txt");
			outputFile.close();
		}
	}

	// Set the monitor to compute the xenon fluence and the retention
	// for the retention calculation
	if (flagXeRetention) {
		// Get the da from ts
		DM da;
		ierr = TSGetDM(ts, &da);
		checkPetscError(ierr, "setupPetsc2DMonitor: TSGetDM failed.");
		// Get the local boundaries
		PetscInt xm, ym;
		ierr = DMDAGetCorners(da, NULL, NULL, NULL, &xm, &ym, NULL);
		checkPetscError(ierr, "setupPetsc2DMonitor: DMDAGetCorners failed.");
		// Create the local vectors on each process
		solverHandler.createLocalNE(xm, ym);

		// Get the previous time if concentrations were stored and initialize
		// the fluence
		if (hasConcentrations) {
			assert(lastTsGroup);

			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			// Initialize the fluence
			auto fluxHandler = solverHandler.getFluxHandler();
			// Increment the fluence with the value at this current timestep
			fluxHandler->computeFluence(previousTime);
		}

		// computeFluence will be called at each timestep
		ierr = TSMonitorSet(ts, computeFluence, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc2DMonitor: TSMonitorSet (computeFluence) failed.");

		// computeXenonRetention2D will be called at each timestep
		ierr = TSMonitorSet(ts, computeXenonRetention2D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: TSMonitorSet (computeXenonRetention2D) "
			"failed.");

		// Master process
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("retentionOut.txt");
			outputFile.close();
		}
	}

	// Set the monitor to save surface plots of clusters concentration
	if (flag2DPlot) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a SurfacePlot
			surfacePlot2D = vizHandlerRegistry->getPlot(
				"surfacePlot2D", viz::PlotType::SURFACE);

			// Create and set the label provider
			auto labelProvider =
				std::make_shared<viz::LabelProvider>("labelProvider");
			labelProvider->axis1Label = "Depth (nm)";
			labelProvider->axis2Label = "Y (nm)";
			labelProvider->axis3Label = "Concentration";

			// Give it to the plot
			surfacePlot2D->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider =
				std::make_shared<viz::dataprovider::CvsXYDataProvider>(
					"dataProvider");

			// Give it to the plot
			surfacePlot2D->setDataProvider(dataProvider);
		}

		// monitorSurface2D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorSurface2D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: TSMonitorSet (monitorSurface2D) failed.");
	}

	// Set the monitor to monitor the concentration of the largest cluster
	if (flagLargest) {
		// Look for the largest cluster
		auto& network = solverHandler.getNetwork();
		largestClusterId2D = network.getLargestClusterId();

		// Find the threshold
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-largest_conc", &largestThreshold2D, &flag);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: PetscOptionsGetReal (-largest_conc) failed.");

		// monitorLargest2D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorLargest2D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc2DMonitor: TSMonitorSet (monitorLargest2D) failed.");
	}

	// Set the monitor to simply change the previous time to the new time
	// monitorTime will be called at each timestep
	ierr = TSMonitorSet(ts, monitorTime, NULL, NULL);
	checkPetscError(
		ierr, "setupPetsc2DMonitor: TSMonitorSet (monitorTime) failed.");

	PetscFunctionReturn(0);
}

/**
 * This operation resets all the global variables to their original values.
 * @return A standard PETSc error code
 */
PetscErrorCode
reset2DMonitor()
{
	timeStepThreshold = 0.0;
	hdf5Stride2D = 0.0;
	hdf5Previous2D = 0;
	hdf5OutputName2D = "xolotlStop.h5";
	previousIFlux2D.clear();
	nInterstitial2D.clear();
	previousHeFlux2D.clear();
	nHelium2D.clear();
	previousDFlux2D.clear();
	nDeuterium2D.clear();
	previousTFlux2D.clear();
	nTritium2D.clear();
	sputteringYield2D = 0.0;
	depthPositions2D.clear();
	iClusterIds2D.clear();
	largestClusterId2D = -1;
	largestThreshold2D = 1.0e-12;

	PetscFunctionReturn(0);
}

} /* end namespace monitor */
} /* end namespace solver */
} /* end namespace xolotl */
