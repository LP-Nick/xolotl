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
#include <xolotl/core/network/AlloyReactionNetwork.h>
#include <xolotl/core/network/IPSIReactionNetwork.h>
#include <xolotl/core/network/NEReactionNetwork.h>
#include <xolotl/io/XFile.h>
#include <xolotl/perf/PerfHandlerRegistry.h>
#include <xolotl/perf/ScopedTimer.h>
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

//! The pointer to the plot used in monitorScatter1D.
std::shared_ptr<viz::IPlot> scatterPlot1D;
//! The pointer to the series plot used in monitorSeries1D.
std::shared_ptr<viz::IPlot> seriesPlot1D;
//! The pointer to the 2D plot used in MonitorSurface.
std::shared_ptr<viz::IPlot> surfacePlot1D;
//! The variable to store the particle flux at the previous time step.
std::vector<double> previousSurfFlux1D, previousBulkFlux1D;
double previousIEventFlux1D = 0.0;
//! The variable to store the total number of atoms going through the surface or
//! bottom.
std::vector<double> nSurf1D, nBulk1D;
double nInterEvent1D = 0.0, nHeliumBurst1D = 0.0, nDeuteriumBurst1D = 0.0,
	   nTritiumBurst1D = 0.0;
//! The variable to store the xenon flux at the previous time step.
double previousXeFlux1D = 0.0;
//! The variable to store the sputtering yield at the surface.
double sputteringYield1D = 0.0;
//! The threshold for the negative concentration
double negThreshold1D = 0.0;
//! How often HDF5 file is written
PetscReal hdf5Stride1D = 0.0;
//! Previous time for HDF5
PetscInt hdf5Previous1D = 0;
//! HDF5 output file name
std::string hdf5OutputName1D = "xolotlStop.h5";
// The vector of depths at which bursting happens
std::vector<PetscInt> depthPositions1D;
// The vector of ids for diffusing interstitial clusters
std::vector<IdType> iClusterIds1D;
// Tracks the previous TS number
PetscInt previousTSNumber1D = -1;
// The id of the largest cluster
int largestClusterId1D = -1;
// The concentration threshold for the largest cluster
double largestThreshold1D = 1.0e-12;

// Timers
std::shared_ptr<perf::ITimer> initTimer;
std::shared_ptr<perf::ITimer> checkNegativeTimer;
std::shared_ptr<perf::ITimer> tridynTimer;
std::shared_ptr<perf::ITimer> startStopTimer;
std::shared_ptr<perf::ITimer> heRetentionTimer;
std::shared_ptr<perf::ITimer> xeRetentionTimer;
std::shared_ptr<perf::ITimer> scatterTimer;
std::shared_ptr<perf::ITimer> seriesTimer;
std::shared_ptr<perf::ITimer> eventFuncTimer;
std::shared_ptr<perf::ITimer> postEventFuncTimer;

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "checkNegative1D")
/**
 * This is a monitoring method that looks at if there are negative
 * concentrations at each time step.
 */
PetscErrorCode
checkNegative1D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(checkNegativeTimer);

	// Initial declaration
	PetscErrorCode ierr;
	double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the network and dof
	auto& network = solverHandler.getNetwork();
	const auto nClusters = network.getNumClusters();

	// Loop on the local grid
	for (auto i = xs; i < xs + xm; i++) {
		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[i]; // Loop on the concentrations
		for (auto l = 0; l < nClusters; l++) {
			if (gridPointSolution[l] < negThreshold1D &&
				gridPointSolution[l] > 0.0) {
				gridPointSolution[l] = negThreshold1D;
			}
			else if (gridPointSolution[l] > -negThreshold1D &&
				gridPointSolution[l] < 0.0) {
				gridPointSolution[l] = -negThreshold1D;
			}
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorLargest1D")
/**
 * This is a monitoring method that looks at the largest cluster concentration
 */
PetscErrorCode
monitorLargest1D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	// Initial declaration
	PetscErrorCode ierr;
	double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Loop on the local grid
	for (auto i = xs; i < xs + xm; i++) {
		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[i];
		// Check the concentration
		if (gridPointSolution[largestClusterId1D] > largestThreshold1D) {
			ierr = TSSetConvergedReason(ts, TS_CONVERGED_USER);
			CHKERRQ(ierr);
			// Send an error
			throw std::runtime_error(
				"\nxolotlSolver::Monitor1D: The largest cluster "
				"concentration is too high!!");
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeTRIDYN1D")
/**
 * This is a monitoring method that will compute the data to send to TRIDYN
 */
PetscErrorCode
computeTRIDYN1D(
	TS ts, PetscInt timestep, PetscReal time, Vec solution, void* ictx)
{
	perf::ScopedTimer myTimer(tridynTimer);

	// Initial declarations
	PetscErrorCode ierr;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the MPI communicator
	auto xolotlComm = util::getMPIComm();

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the network
	auto& network = solverHandler.getNetwork();
	const auto dof = network.getDOF();
	const auto numSpecies = network.getSpeciesListSize();

	// Get the position of the surface
	auto surfacePos = static_cast<PetscInt>(solverHandler.getSurfacePosition());

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// Get the array of concentration
	PetscReal** solutionArray;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Save current concentrations as an HDF5 file.
	//
	// First create the file for parallel file access.
	std::ostringstream tdFileStr;
	tdFileStr << "TRIDYN_" << timestep << ".h5";
	io::HDF5File tdFile(tdFileStr.str(),
		io::HDF5File::AccessMode::CreateOrTruncateIfExists, xolotlComm, true);

	// Define a dataset for concentrations.
	// Everyone must create the dataset with the same shape.
	const auto numValsPerGridpoint = 5 + 2;
	const auto firstIdxToWrite = (surfacePos + solverHandler.getLeftOffset());
	const auto numGridpointsWithConcs = (Mx - firstIdxToWrite);
	io::HDF5File::SimpleDataSpace<2>::Dimensions concsDsetDims = {
		(hsize_t)numGridpointsWithConcs, numValsPerGridpoint};
	io::HDF5File::SimpleDataSpace<2> concsDsetSpace(concsDsetDims);

	const std::string concsDsetName = "concs";
	io::HDF5File::DataSet<double> concsDset(
		tdFile, concsDsetName, concsDsetSpace);

	// Specify the concentrations we will write.
	// We only consider our own grid points.
	const auto myFirstIdxToWrite =
		std::max((IdType)xs, (IdType)firstIdxToWrite);
	auto myEndIdx = (xs + xm); // "end" in the C++ sense; i.e., one-past-last
	auto myNumPointsToWrite =
		(myEndIdx > myFirstIdxToWrite) ? (myEndIdx - myFirstIdxToWrite) : 0;
	io::HDF5File::DataSet<double>::DataType2D<numValsPerGridpoint> myConcs(
		myNumPointsToWrite);

	for (auto xi = myFirstIdxToWrite; xi < myEndIdx; ++xi) {
		if (xi >= firstIdxToWrite) {
			// Determine current gridpoint value.
			double x = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];

			// Access the solution data for this grid point.
			auto gridPointSolution = solutionArray[xi];
			using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
				Kokkos::MemoryUnmanaged>;
			auto hConcs = HostUnmanaged(gridPointSolution, dof);
			auto dConcs = Kokkos::View<double*>("Concentrations", dof);
			deep_copy(dConcs, hConcs);

			// Get the total concentrations at this grid point
			auto currIdx = (PetscInt)xi - myFirstIdxToWrite;
			myConcs[currIdx][0] = (x - (grid[surfacePos + 1] - grid[1]));
			// Get the total concentrations at this grid point
			for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
				myConcs[currIdx][id() + 1] +=
					network.getTotalAtomConcentration(dConcs, id, 1);
			}
			myConcs[currIdx][6] = gridPointSolution[dof];
		}
	}

	// Write the concs dataset in parallel.
	// (We write only our part.)
	concsDset.parWrite2D<numValsPerGridpoint>(
		xolotlComm, myFirstIdxToWrite - firstIdxToWrite, myConcs);

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "startStop1D")
/**
 * This is a monitoring method that update an hdf5 file every given time.
 */
PetscErrorCode
startStop1D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(startStopTimer);

	// Initial declaration
	PetscErrorCode ierr;
	const double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Compute the dt
	double previousTime = solverHandler.getPreviousTime();
	double dt = time - previousTime;

	// Don't do anything if it is not on the stride
	if (((PetscInt)((time + dt / 10.0) / hdf5Stride1D) <= hdf5Previous1D) &&
		timestep > 0) {
		PetscFunctionReturn(0);
	}

	// Update the previous time
	if ((PetscInt)((time + dt / 10.0) / hdf5Stride1D) > hdf5Previous1D)
		hdf5Previous1D++;

	// Gets MPI comm
	auto xolotlComm = util::getMPIComm();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the network and dof
	auto& network = solverHandler.getNetwork();
	const auto dof = network.getDOF();

	// Create an array for the concentration
	double concArray[dof + 1][2];

	// Get the position of the surface
	auto surfacePos = solverHandler.getSurfacePosition();

	// Open the existing HDF5 file
	io::XFile checkpointFile(
		hdf5OutputName1D, xolotlComm, io::XFile::AccessMode::OpenReadWrite);

	// Get the current time step
	double currentTimeStep;
	ierr = TSGetTimeStep(ts, &currentTimeStep);
	CHKERRQ(ierr);

	// Add a concentration time step group for the current time step.
	auto concGroup = checkpointFile.getGroup<io::XFile::ConcentrationGroup>();
	assert(concGroup);
	auto tsGroup = concGroup->addTimestepGroup(
		timestep, time, previousTime, currentTimeStep);

	// Get the names of the species in the network
	auto numSpecies = network.getSpeciesListSize();
	std::vector<std::string> names;
	for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
		names.push_back(network.getSpeciesName(id));
	}

	if (solverHandler.moveSurface() || solverHandler.getLeftOffset() == 1) {
		// Write the surface positions and the associated interstitial
		// quantities in the concentration sub group
		tsGroup->writeSurface1D(surfacePos, nInterEvent1D, previousIEventFlux1D,
			nSurf1D, previousSurfFlux1D, names);
	}

	// Write the bottom impurity information if the bottom is a free surface
	if (solverHandler.getRightOffset() == 1)
		tsGroup->writeBottom1D(nBulk1D, previousBulkFlux1D, names);

	// Write the bursting information if the bubble bursting is used
	if (solverHandler.burstBubbles())
		tsGroup->writeBursting1D(
			nHeliumBurst1D, nDeuteriumBurst1D, nTritiumBurst1D);

	// Determine the concentration values we will write.
	// We only examine and collect the grid points we own.
	// TODO measure impact of us building the flattened representation
	// rather than a ragged 2D representation.
	io::XFile::TimestepGroup::Concs1DType concs(xm);
	for (auto i = 0; i < xm; ++i) {
		// Access the solution data for the current grid point.
		auto gridPointSolution = solutionArray[xs + i];

		for (auto l = 0; l < dof + 1; ++l) {
			if (std::fabs(gridPointSolution[l]) > 1.0e-16) {
				concs[i].emplace_back(l, gridPointSolution[l]);
			}
		}
	}

	// Write our concentration data to the current timestep group
	// in the HDF5 file.
	// We only write the data for the grid points we own.
	tsGroup->writeConcentrations(checkpointFile, xs, concs);

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	ierr = computeTRIDYN1D(ts, timestep, time, solution, NULL);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeHeliumRetention1D")
/**
 * This is a monitoring method that will compute the helium retention
 */
PetscErrorCode
computeHeliumRetention1D(TS ts, PetscInt, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(heRetentionTimer);

	// Initial declarations
	PetscErrorCode ierr;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the flux handler that will be used to know the fluence
	auto fluxHandler = solverHandler.getFluxHandler();
	// Get the diffusion handler
	auto diffusionHandler = solverHandler.getDiffusionHandler();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the physical grid
	auto grid = solverHandler.getXGrid();
	// Get the position of the surface
	auto surfacePos = solverHandler.getSurfacePosition();

	// Get the network
	using NetworkType = core::network::IPSIReactionNetwork;
	using AmountType = NetworkType::AmountType;
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	const auto dof = network.getDOF();

	// Get the array of concentration
	PetscReal** solutionArray;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Store the concentration over the grid
	auto numSpecies = network.getSpeciesListSize();
	auto myConcData = std::vector<double>(numSpecies, 0.0);

	// Declare the pointer for the concentrations at a specific grid point
	PetscReal* gridPointSolution;

	// Loop on the grid
	for (auto xi = xs; xi < xs + xm; xi++) {
		// Boundary conditions
		if (xi < surfacePos + solverHandler.getLeftOffset() ||
			xi >= Mx - solverHandler.getRightOffset())
			continue;

		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[xi];

		double hx = grid[xi + 1] - grid[xi];

		using HostUnmanaged =
			Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
		auto hConcs = HostUnmanaged(gridPointSolution, dof);
		auto dConcs = Kokkos::View<double*>("Concentrations", dof);
		deep_copy(dConcs, hConcs);

		// Get the total concentrations at this grid point
		for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
			myConcData[id()] +=
				network.getTotalAtomConcentration(dConcs, id, 1) * hx;
		}
	}

	// Get the current process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Determine total concentrations for He, D, T.
	auto totalConcData = std::vector<double>(numSpecies, 0.0);

	MPI_Reduce(myConcData.data(), totalConcData.data(), numSpecies, MPI_DOUBLE,
		MPI_SUM, 0, xolotlComm);

	// Get the delta time from the previous timestep to this timestep
	double previousTime = solverHandler.getPreviousTime();
	double dt = time - previousTime;

	// Look at the fluxes leaving the free surface
	if (solverHandler.getLeftOffset() == 1) {
		// Set the surface position
		auto xi = surfacePos + 1;

		// Value to know on which processor is the surface
		int surfaceProc = 0;

		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm) {
			// Compute the total number of impurities that left at the surface
			for (auto i = 0; i < numSpecies; ++i) {
				nSurf1D[i] += previousSurfFlux1D[i] * dt;
			}
			auto myFluxData = std::vector<double>(numSpecies, 0.0);

			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[xi];

			// Factor for finite difference
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < Mx) {
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
			double factor = 2.0 / (hxLeft + hxRight);

			// Get the vector of diffusing clusters
			auto diffusingIds = diffusionHandler->getDiffusingIds();

			network.updateOutgoingDiffFluxes(
				gridPointSolution, factor, diffusingIds, myFluxData, xi - xs);

			// Take into account the surface advection
			// Get the surface advection handler
			auto advecHandler = solverHandler.getAdvectionHandler();
			// Get the sink strengths and advecting clusters
			auto sinkStrengths = advecHandler->getSinkStrengths();
			auto advecClusters = advecHandler->getAdvectingClusters();
			// Set the distance from the surface
			double distance = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1] -
				advecHandler->getLocation();

			network.updateOutgoingAdvecFluxes(gridPointSolution,
				3.0 /
					(core::kBoltzmann * distance * distance * distance *
						distance),
				advecClusters, sinkStrengths, myFluxData, xi - xs);

			for (auto i = 0; i < numSpecies; ++i) {
				previousSurfFlux1D[i] = myFluxData[i];
			}

			// Set the surface processor
			surfaceProc = procId;
		}

		// Get which processor will send the information
		// TODO do we need to do this allreduce just to figure out
		// who owns the data?
		// And is it supposed to be a sum?   Why not a min?
		int surfaceId = 0;
		MPI_Allreduce(
			&surfaceProc, &surfaceId, 1, MPI_INT, MPI_SUM, xolotlComm);

		// Send the information about impurities
		// to the other processes
		std::vector<double> countFluxData;
		for (auto i = 0; i < numSpecies; ++i) {
			countFluxData.push_back(nSurf1D[i]);
			countFluxData.push_back(previousSurfFlux1D[i]);
		}
		MPI_Bcast(countFluxData.data(), countFluxData.size(), MPI_DOUBLE,
			surfaceId, xolotlComm);

		// Extract impurity data from broadcast buffer.
		for (auto i = 0; i < numSpecies; ++i) {
			nSurf1D[i] = countFluxData[2 * i];
			previousSurfFlux1D[i] = countFluxData[(2 * i) + 1];
		}
	}

	// Look at the fluxes going in the bulk if the bottom is a free surface
	if (solverHandler.getRightOffset() == 1) {
		// Set the bottom surface position
		auto xi = Mx - 2;

		// Value to know on which processor is the bottom
		int bottomProc = 0;

		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm) {
			// Compute the total number of impurities that went in the bulk
			for (auto i = 0; i < numSpecies; ++i) {
				nBulk1D[i] += previousBulkFlux1D[i] * dt;
			}
			auto myFluxData = std::vector<double>(numSpecies, 0.0);

			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[xi];

			// Factor for finite difference
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < Mx) {
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
			double factor = 2.0 / (hxLeft + hxRight);

			// Get the vector of diffusing clusters
			auto diffusingIds = diffusionHandler->getDiffusingIds();

			network.updateOutgoingDiffFluxes(
				gridPointSolution, factor, diffusingIds, myFluxData, xi - xs);

			for (auto i = 0; i < numSpecies; ++i) {
				previousBulkFlux1D[i] = myFluxData[i];
			}

			// Set the bottom processor
			bottomProc = procId;
		}

		// Get which processor will send the information
		// TODO do we need to do this allreduce just to figure out
		// who owns the data?
		// And is it supposed to be a sum?   Why not a min?
		int bottomId = 0;
		MPI_Allreduce(&bottomProc, &bottomId, 1, MPI_INT, MPI_SUM, xolotlComm);

		// Send the information about impurities
		// to the other processes
		std::vector<double> countFluxData;
		for (auto i = 0; i < numSpecies; ++i) {
			countFluxData.push_back(nBulk1D[i]);
			countFluxData.push_back(previousBulkFlux1D[i]);
		}
		MPI_Bcast(countFluxData.data(), countFluxData.size(), MPI_DOUBLE,
			bottomId, xolotlComm);

		// Extract impurity data from broadcast buffer.
		for (auto i = 0; i < numSpecies; ++i) {
			nBulk1D[i] = countFluxData[2 * i];
			previousBulkFlux1D[i] = countFluxData[(2 * i) + 1];
		}
	}

	// Master process
	if (procId == 0) {
		// Get the fluence
		double fluence = fluxHandler->getFluence();

		// Print the result
		std::cout << "\nTime: " << time << std::endl;
		for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
			std::cout << network.getSpeciesName(id)
					  << " content = " << totalConcData[id()] << '\n';
		}
		std::cout << "Fluence = " << fluence << '\n' << std::endl;

		// Uncomment to write the retention and the fluence in a file
		std::ofstream outputFile;
		outputFile.open("retentionOut.txt", std::ios::app);
		outputFile << fluence << ' ';
		for (auto i = 0; i < numSpecies; ++i) {
			outputFile << totalConcData[i] << ' ';
		}
		if (solverHandler.getRightOffset() == 1) {
			for (auto i = 0; i < numSpecies; ++i) {
				outputFile << nBulk1D[i] << ' ';
			}
		}
		if (solverHandler.getLeftOffset() == 1) {
			for (auto i = 0; i < numSpecies; ++i) {
				outputFile << nSurf1D[i] << ' ';
			}
		}
		outputFile << nHeliumBurst1D << " " << nDeuteriumBurst1D << " "
				   << nTritiumBurst1D << std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeXenonRetention1D")
/**
 * This is a monitoring method that will compute the xenon retention
 */
PetscErrorCode
computeXenonRetention1D(TS ts, PetscInt, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(xeRetentionTimer);

	// Initial declarations
	PetscErrorCode ierr;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	using NetworkType = core::network::NEReactionNetwork;
	using Spec = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;

	// Degrees of freedom is the total number of clusters in the network
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	const auto dof = network.getDOF();

	// Get the complete data array, including ghost cells
	Vec localSolution;
	ierr = DMGetLocalVector(da, &localSolution);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, solution, INSERT_VALUES, localSolution);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, solution, INSERT_VALUES, localSolution);
	CHKERRQ(ierr);
	// Get the array of concentration
	PetscReal** solutionArray;
	ierr = DMDAVecGetArrayDOFRead(da, localSolution, &solutionArray);
	CHKERRQ(ierr);

	// Store the concentration and other values over the grid
	double xeConcentration = 0.0, bubbleConcentration = 0.0, radii = 0.0,
		   partialBubbleConcentration = 0.0, partialRadii = 0.0,
		   partialSize = 0.0;

	// Declare the pointer for the concentrations at a specific grid point
	PetscReal* gridPointSolution;

	// Get the minimum size for the radius
	auto minSizes = solverHandler.getMinSizes();

	// Get Xe_1
	Composition xeComp = Composition::zero();
	xeComp[Spec::Xe] = 1;
	auto xeCluster = network.findCluster(xeComp, plsm::onHost);
	auto xeId = xeCluster.getId();

	// Loop on the grid
	for (auto xi = xs; xi < xs + xm; xi++) {
		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[xi];

		using HostUnmanaged =
			Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
		auto hConcs = HostUnmanaged(gridPointSolution, dof);
		auto dConcs = Kokkos::View<double*>("Concentrations", dof);
		deep_copy(dConcs, hConcs);

		// Initialize the volume fraction and hx
		double hx = grid[xi + 1] - grid[xi];

		// Get the concentrations
		xeConcentration +=
			network.getTotalAtomConcentration(dConcs, Spec::Xe, 1) * hx;
		bubbleConcentration +=
			network.getTotalConcentration(dConcs, Spec::Xe, 1) * hx;
		radii += network.getTotalRadiusConcentration(dConcs, Spec::Xe, 1) * hx;
		partialBubbleConcentration =
			network.getTotalConcentration(dConcs, Spec::Xe, minSizes[0]) * hx;
		partialRadii +=
			network.getTotalRadiusConcentration(dConcs, Spec::Xe, minSizes[0]) *
			hx;
		partialSize +=
			network.getTotalAtomConcentration(dConcs, Spec::Xe, minSizes[0]) *
			hx;

		// Set the volume fraction
		double volumeFrac =
			network.getTotalVolumeFraction(dConcs, Spec::Xe, minSizes[0]);
		solverHandler.setVolumeFraction(volumeFrac, xi - xs);
		// Set the monomer concentration
		solverHandler.setMonomerConc(
			gridPointSolution[xeCluster.getId()], xi - xs);
	}

	// Get the current process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Sum all the concentrations through MPI reduce
	std::array<double, 6> myConcData{xeConcentration, bubbleConcentration,
		radii, partialBubbleConcentration, partialRadii, partialSize};
	std::array<double, 6> totalConcData{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	MPI_Reduce(myConcData.data(), totalConcData.data(), myConcData.size(),
		MPI_DOUBLE, MPI_SUM, 0, xolotlComm);

	// GB
	// Get the delta time from the previous timestep to this timestep
	double dt = time - solverHandler.getPreviousTime();
	// Sum and gather the previous flux
	double globalXeFlux = 0.0;
	// Get the vector from the solver handler
	auto gbVector = solverHandler.getGBVector();
	// Get the previous flux vector
	auto& localNE = solverHandler.getLocalNE();
	// Loop on the GB
	for (auto const& pair : gbVector) {
		// Middle
		auto xi = std::get<0>(pair);
		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm) {
			double previousXeFlux = std::get<1>(localNE[xi - xs][0][0]);
			globalXeFlux += previousXeFlux * (grid[xi + 1] - grid[xi]);
			// Set the amount in the vector we keep
			solverHandler.setLocalXeRate(previousXeFlux * dt, xi - xs);
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

	// Get the number of species
	auto numSpecies = network.getSpeciesListSize();

	// Get the vector of diffusing clusters
	auto diffusionHandler = solverHandler.getDiffusionHandler();
	auto diffusingIds = diffusionHandler->getDiffusingIds();

	// Loop on the GB
	for (auto const& pair : gbVector) {
		// Local rate
		auto myRate = std::vector<double>(numSpecies, 0.0);
		// Define left and right with reference to the middle point
		// Middle
		auto xi = std::get<0>(pair);

		// Factor for finite difference
		double hxLeft = 0.0, hxRight = 0.0;
		if (xi >= 1 && xi < Mx) {
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
		double factor = 2.0 / (hxLeft + hxRight);
		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm) {
			// Left
			xi = std::get<0>(pair) - 1;
			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[xi];
			// Compute the flux coming from the left
			network.updateOutgoingDiffFluxes(gridPointSolution, factor / hxLeft,
				diffusingIds, myRate, xi + 1 - xs);

			// Right
			xi = std::get<0>(pair) + 1;
			gridPointSolution = solutionArray[xi];
			// Compute the flux coming from the right
			network.updateOutgoingDiffFluxes(gridPointSolution,
				factor / hxRight, diffusingIds, myRate, xi + 1 - xs);

			// Middle
			xi = std::get<0>(pair);
			solverHandler.setPreviousXeFlux(myRate[0], xi - xs);
		}
	}

	// Master process
	if (procId == 0) {
		// Get the number of xenon that went to the GB
		double nXenon = solverHandler.getNXeGB();

		// Print the result
		std::cout << "\nTime: " << time << std::endl;
		std::cout << "Xenon concentration = " << totalConcData[0] << std::endl;
		std::cout << "Xenon GB = " << nXenon << std::endl << std::endl;

		// Make sure the average partial radius makes sense
		double averagePartialRadius = 0.0, averagePartialSize = 0.0;
		if (totalConcData[3] > 1.e-16) {
			averagePartialRadius = totalConcData[4] / totalConcData[3];
			averagePartialSize = totalConcData[5] / totalConcData[3];
		}

		// Uncomment to write the content in a file
		std::ofstream outputFile;
		outputFile.open("retentionOut.txt", std::ios::app);
		outputFile << time << " " << totalConcData[0] << " "
				   << totalConcData[2] / totalConcData[1] << " "
				   << averagePartialRadius << " " << totalConcData[3] << " "
				   << averagePartialSize << std::endl;
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
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "profileTemperature1D")
/**
 * This is a monitoring method that will store the temperature profile
 */
PetscErrorCode
profileTemperature1D(
	TS ts, PetscInt timestep, PetscReal time, Vec solution, void* ictx)
{
	// Initial declarations
	PetscErrorCode ierr;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Gets the process ID (important when it is running in parallel)
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the network and dof
	auto& network = solverHandler.getNetwork();
	const auto dof = network.getDOF();

	// Get the position of the surface
	auto surfacePos = solverHandler.getSurfacePosition();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// Get the array of concentration
	PetscReal** solutionArray;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Declare the pointer for the concentrations at a specific grid point
	PetscReal* gridPointSolution;

	// Create the output file
	std::ofstream outputFile;
	if (procId == 0) {
		outputFile.open("tempProf.txt", std::ios::app);
		outputFile << time;
	}

	// Loop on the entire grid
	for (auto xi = surfacePos + solverHandler.getLeftOffset();
		 xi < Mx - solverHandler.getRightOffset(); xi++) {
		// Set x
		double x = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];

		double localTemp = 0.0;
		// Check if this process is in charge of xi
		if (xi >= xs && xi < xs + xm) {
			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[xi];

			// Get the local temperature
			localTemp = gridPointSolution[dof];
		}

		// Get the value on procId = 0
		double temperature = 0.0;
		MPI_Reduce(
			&localTemp, &temperature, 1, MPI_DOUBLE, MPI_SUM, 0, xolotlComm);

		// The master process writes in the file
		if (procId == 0) {
			outputFile << " " << temperature;
		}
	}

	// Close the file
	if (procId == 0) {
		outputFile << std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeAlloy1D")
/**
 * This is a monitoring method that will compute average density and diameter
 * of defects
 */
PetscErrorCode
computeAlloy1D(
	TS ts, PetscInt timestep, PetscReal time, Vec solution, void* ictx)
{
	// Initial declarations
	PetscErrorCode ierr;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the MPI comm
	auto xolotlComm = util::getMPIComm();

	// Get the process ID
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the physical grid and its length
	auto grid = solverHandler.getXGrid();
	auto xSize = grid.size();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the array of concentration
	PetscReal** solutionArray;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	using NetworkType = core::network::AlloyReactionNetwork;
	using Spec = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;

	// Degrees of freedom is the total number of clusters in the network
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	const auto dof = network.getDOF();
	auto numSpecies = network.getSpeciesListSize();
	auto myData = std::vector<double>(numSpecies * 4, 0.0);

	// Get the minimum size for the loop densities and diameters
	auto minSizes = solverHandler.getMinSizes();

	// Declare the pointer for the concentrations at a specific grid point
	PetscReal* gridPointSolution;

	// Get the position of the surface
	auto surfacePos = solverHandler.getSurfacePosition();

	// Loop on the grid
	for (auto xi = xs; xi < xs + xm; xi++) {
		// Boundary conditions
		if (xi < surfacePos + solverHandler.getLeftOffset() ||
			xi == Mx - solverHandler.getRightOffset())
			continue;

		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[xi];

		using HostUnmanaged =
			Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
		auto hConcs = HostUnmanaged(gridPointSolution, dof);
		auto dConcs = Kokkos::View<double*>("Concentrations", dof);
		deep_copy(dConcs, hConcs);

		// Loop on the species
		for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
			myData[4 * id()] += network.getTotalConcentration(dConcs, id, 1);
			myData[(4 * id()) + 1] +=
				2.0 * network.getTotalRadiusConcentration(dConcs, id, 1);
			myData[(4 * id()) + 2] +=
				network.getTotalConcentration(dConcs, id, minSizes[id()]);
			myData[(4 * id()) + 3] += 2.0 *
				network.getTotalRadiusConcentration(dConcs, id, minSizes[id()]);
		}
	}

	// Sum all the concentrations through MPI reduce
	auto globalData = std::vector<double>(myData.size(), 0.0);
	MPI_Reduce(myData.data(), globalData.data(), myData.size(), MPI_DOUBLE,
		MPI_SUM, 0, xolotlComm);

	// Average the data
	if (procId == 0) {
		for (auto i = 0; i < numSpecies; ++i) {
			globalData[(4 * i) + 1] /= globalData[4 * i];
			globalData[(4 * i) + 3] /= globalData[(4 * i) + 2];
			globalData[4 * i] /= (grid[Mx] - grid[surfacePos + 1]);
			globalData[(4 * i) + 2] /= (grid[Mx] - grid[surfacePos + 1]);
		}

		// Set the output precision
		const int outputPrecision = 5;

		// Open the output file
		std::fstream outputFile;
		outputFile.open("Alloy.dat", std::fstream::out | std::fstream::app);
		outputFile << std::setprecision(outputPrecision);

		// Output the data
		outputFile << timestep << " " << time << " ";
		for (auto i = 0; i < numSpecies; ++i) {
			outputFile << globalData[i * 4] << " " << globalData[(i * 4) + 1]
					   << " " << globalData[(i * 4) + 2] << " "
					   << globalData[(i * 4) + 3] << " ";
		}
		outputFile << std::endl;

		// Close the output file
		outputFile.close();
	}

	// Restore the PETSC solution array
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorScatter1D")
/**
 * This is a monitoring method that will save 1D plots of the xenon
 * concentration distribution at the middle of the grid.
 */
PetscErrorCode
monitorScatter1D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(scatterTimer);

	// Initial declarations
	PetscErrorCode ierr;
	double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Don't do anything if it is not on the stride
	if (timestep % 200 != 0)
		PetscFunctionReturn(0);

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

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the network and its size
	using NetworkType = core::network::NEReactionNetwork;
	using Spec = typename NetworkType::Species;
	using Region = typename NetworkType::Region;
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	auto networkSize = network.getNumClusters();

	// Get the index of the middle of the grid
	auto ix = Mx / 2;

	// If the middle is on this process
	if (ix >= xs && ix < xs + xm) {
		// Create a DataPoint vector to store the data to give to the data
		// provider for the visualization
		auto myPoints =
			std::make_shared<std::vector<viz::dataprovider::DataPoint>>();

		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[ix];

		for (auto i = 0; i < networkSize; i++) {
			// Create a Point with the concentration[i] as the value
			// and add it to myPoints
			auto cluster = network.getCluster(i);
			const Region& clReg = cluster.getRegion();
			for (auto j : makeIntervalRange(clReg[Spec::Xe])) {
				viz::dataprovider::DataPoint aPoint;
				aPoint.value = gridPointSolution[i];
				aPoint.t = time;
				aPoint.x = (double)j;
				myPoints->push_back(aPoint);
			}
		}

		// Get the data provider and give it the points
		scatterPlot1D->getDataProvider()->setDataPoints(myPoints);

		// Change the title of the plot and the name of the data
		std::stringstream title;
		title << "Size Distribution";
		scatterPlot1D->getDataProvider()->setDataName(title.str());
		scatterPlot1D->plotLabelProvider->titleLabel = title.str();
		// Give the time to the label provider
		std::stringstream timeLabel;
		timeLabel << "time: " << std::setprecision(4) << time << "s";
		scatterPlot1D->plotLabelProvider->timeLabel = timeLabel.str();
		// Get the current time step
		PetscReal currentTimeStep;
		ierr = TSGetTimeStep(ts, &currentTimeStep);
		CHKERRQ(ierr);
		// Give the timestep to the label provider
		std::stringstream timeStepLabel;
		timeStepLabel << "dt: " << std::setprecision(4) << currentTimeStep
					  << "s";
		scatterPlot1D->plotLabelProvider->timeStepLabel = timeStepLabel.str();

		// Render and save in file
		std::stringstream fileName;
		fileName << "Scatter_TS" << timestep << ".png";
		scatterPlot1D->write(fileName.str());
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorSeries1D")
/**
 * This is a monitoring method that will save 1D plots of many concentrations
 */
PetscErrorCode
monitorSeries1D(TS ts, PetscInt timestep, PetscReal time, Vec solution, void*)
{
	perf::ScopedTimer myTimer(seriesTimer);

	// Initial declarations
	PetscErrorCode ierr;
	const double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;
	double x = 0.0;

	PetscFunctionBeginUser;

	// Don't do anything if it is not on the stride
	if (timestep % 10 != 0)
		PetscFunctionReturn(0);

	// Get the number of processes
	auto xolotlComm = util::getMPIComm();
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
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the network and its size
	auto& network = solverHandler.getNetwork();
	const auto networkSize = network.getNumClusters();

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// To plot a maximum of 18 clusters of the whole benchmark
	const auto loopSize = std::min(18, (int)networkSize);

	if (procId == 0) {
		// Create a DataPoint vector to store the data to give to the data
		// provider for the visualization
		std::vector<std::vector<viz::dataprovider::DataPoint>> myPoints(
			loopSize);

		// Loop on the grid
		for (auto xi = xs; xi < xs + xm; xi++) {
			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[xi];

			for (auto i = 0; i < loopSize; i++) {
				// Create a DataPoint with the concentration[i] as the value
				// and add it to myPoints
				viz::dataprovider::DataPoint aPoint;
				aPoint.value = gridPointSolution[i];
				aPoint.t = time;
				aPoint.x = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
				myPoints[i].push_back(aPoint);
			}
		}

		// Loop on the other processes
		for (auto i = 1; i < worldSize; i++) {
			// Get the size of the local grid of that process
			int localSize = 0;
			MPI_Recv(
				&localSize, 1, MPI_INT, i, 20, xolotlComm, MPI_STATUS_IGNORE);

			// Loop on their grid
			for (auto k = 0; k < localSize; k++) {
				// Get the position
				MPI_Recv(
					&x, 1, MPI_DOUBLE, i, 21, xolotlComm, MPI_STATUS_IGNORE);

				for (auto j = 0; j < loopSize; j++) {
					// and the concentrations
					double conc = 0.0;
					MPI_Recv(&conc, 1, MPI_DOUBLE, i, 22, xolotlComm,
						MPI_STATUS_IGNORE);

					// Create a Point with the concentration[i] as the value
					// and add it to myPoints
					viz::dataprovider::DataPoint aPoint;
					aPoint.value = conc;
					aPoint.t = time;
					aPoint.x = x;
					myPoints[j].push_back(aPoint);
				}
			}
		}

		for (auto i = 0; i < loopSize; i++) {
			// Get the data provider and give it the points
			auto thePoints =
				std::make_shared<std::vector<viz::dataprovider::DataPoint>>(
					myPoints[i]);
			seriesPlot1D->getDataProvider(i)->setDataPoints(thePoints);
			// TODO: get the name or comp of the cluster
			seriesPlot1D->getDataProvider(i)->setDataName(std::to_string(i));
		}

		// Change the title of the plot
		std::stringstream title;
		title << "Concentrations";
		seriesPlot1D->plotLabelProvider->titleLabel = title.str();
		// Give the time to the label provider
		std::stringstream timeLabel;
		timeLabel << "time: " << std::setprecision(4) << time << "s";
		seriesPlot1D->plotLabelProvider->timeLabel = timeLabel.str();
		// Get the current time step
		PetscReal currentTimeStep;
		ierr = TSGetTimeStep(ts, &currentTimeStep);
		CHKERRQ(ierr);
		// Give the timestep to the label provider
		std::stringstream timeStepLabel;
		timeStepLabel << "dt: " << std::setprecision(4) << currentTimeStep
					  << "s";
		seriesPlot1D->plotLabelProvider->timeStepLabel = timeStepLabel.str();

		// Render and save in file
		std::stringstream fileName;
		fileName << "log_series_TS" << timestep << ".ppm";
		seriesPlot1D->write(fileName.str());
	}

	else {
		// Send the value of the local grid size to the master process
		MPI_Send(&xm, 1, MPI_DOUBLE, 0, 20, xolotlComm);

		// Loop on the grid
		for (auto xi = xs; xi < xs + xm; xi++) {
			// Dump x
			x = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];

			// Get the pointer to the beginning of the solution data for this
			// grid point
			gridPointSolution = solutionArray[xi];

			// Send the value of the local position to the master process
			MPI_Send(&x, 1, MPI_DOUBLE, 0, 21, xolotlComm);

			for (auto i = 0; i < loopSize; i++) {
				// Send the value of the concentrations to the master process
				MPI_Send(
					&gridPointSolution[i], 1, MPI_DOUBLE, 0, 22, xolotlComm);
			}
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "eventFunction1D")
/**
 * This is a method that checks if the surface should move or bursting happen
 */
PetscErrorCode
eventFunction1D(TS ts, PetscReal time, Vec solution, PetscScalar* fvalue, void*)
{
	perf::ScopedTimer myTimer(eventFuncTimer);

	// Initial declaration
	PetscErrorCode ierr;
	double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;
	depthPositions1D.clear();
	fvalue[0] = 1.0, fvalue[1] = 1.0, fvalue[2] = 1.0;

	PetscFunctionBeginUser;

	PetscInt TSNumber = -1;
	ierr = TSGetStepNumber(ts, &TSNumber);

	// Skip if it is the same TS as before
	if (TSNumber == previousTSNumber1D)
		PetscFunctionReturn(0);

	// Set the previous TS number
	previousTSNumber1D = TSNumber;

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

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the position of the surface
	auto surfacePos = solverHandler.getSurfacePosition();
	auto xi = surfacePos + solverHandler.getLeftOffset();

	// Get the network
	using NetworkType = core::network::IPSIReactionNetwork;
	auto& network = dynamic_cast<NetworkType&>(solverHandler.getNetwork());
	// Get the number of species
	auto numSpecies = network.getSpeciesListSize();
	auto specIdI = network.getInterstitialSpeciesId();

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// Get the flux handler to know the flux amplitude.
	auto fluxHandler = solverHandler.getFluxHandler();
	double heliumFluxAmplitude = fluxHandler->getFluxAmplitude();

	// Get the delta time from the previous timestep to this timestep
	double dt = time - solverHandler.getPreviousTime();

	// Work of the moving surface first
	if (solverHandler.moveSurface()) {
		// Write the initial surface position
		if (procId == 0 && util::equal(time, 0.0)) {
			std::ofstream outputFile;
			outputFile.open("surface.txt", std::ios::app);
			outputFile << time << " " << grid[surfacePos + 1] - grid[1]
					   << std::endl;
			outputFile.close();
		}

		// Value to know on which processor is the location of the surface,
		// for MPI usage
		int surfaceProc = 0;

		// if xi is on this process
		if (xi >= xs && xi < xs + xm) {
			// Get the concentrations at xi = surfacePos + 1
			gridPointSolution = solutionArray[xi];

			// Compute the total density of intersitials that escaped from the
			// surface since last timestep using the stored flux
			nInterEvent1D += previousIEventFlux1D * dt;

			// Remove the sputtering yield since last timestep
			nInterEvent1D -= sputteringYield1D * heliumFluxAmplitude * dt;

			// Initialize the value for the flux
			auto myFlux = std::vector<double>(numSpecies, 0.0);

			// Factor for finite difference
			double hxLeft = 0.0, hxRight = 0.0;
			if (xi >= 1 && xi < Mx) {
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
			double factor = 2.0 / (hxLeft + hxRight);

			network.updateOutgoingDiffFluxes(
				gridPointSolution, factor, iClusterIds1D, myFlux, xi - xs);
			// Update the previous flux
			previousIEventFlux1D = myFlux[specIdI()];

			// Set the surface processor
			surfaceProc = procId;
		}

		// Get which processor will send the information
		int surfaceId = 0;
		MPI_Allreduce(
			&surfaceProc, &surfaceId, 1, MPI_INT, MPI_SUM, xolotlComm);

		// Send the information about nInterEvent1D and previousFlux1D
		// to the other processes
		MPI_Bcast(&nInterEvent1D, 1, MPI_DOUBLE, surfaceId, xolotlComm);
		MPI_Bcast(&previousIEventFlux1D, 1, MPI_DOUBLE, surfaceId, xolotlComm);

		// Now that all the processes have the same value of nInterstitials,
		// compare it to the threshold to now if we should move the surface

		// Get the initial vacancy concentration
		double initialVConc = solverHandler.getInitialVConc();

		// The density of tungsten is 62.8 atoms/nm3, thus the threshold is
		double threshold = (62.8 - initialVConc) * (grid[xi] - grid[xi - 1]);
		if (nInterEvent1D > threshold) {
			// The surface is moving
			fvalue[0] = 0;
		}

		// Moving the surface back
		else if (nInterEvent1D < -threshold / 10.0) {
			// The surface is moving
			fvalue[1] = 0;
		}
	}

	// Now work on the bubble bursting
	if (solverHandler.burstBubbles()) {
		using NetworkType = core::network::IPSIReactionNetwork;
		auto psiNetwork = dynamic_cast<NetworkType*>(&network);
		auto dof = network.getDOF();
		auto specIdHe = psiNetwork->getHeliumSpeciesId();

		// Compute the prefactor for the probability (arbitrary)
		double prefactor =
			heliumFluxAmplitude * dt * solverHandler.getBurstingFactor();

		// The depth parameter to know where the bursting should happen
		double depthParam = solverHandler.getTauBursting(); // nm
		// The number of He per V in a bubble
		double heVRatio = solverHandler.getHeVRatio();

		// For now we are not bursting
		bool burst = false;

		// Loop on the full grid of interest
		for (xi = surfacePos + solverHandler.getLeftOffset();
			 xi < Mx - solverHandler.getRightOffset(); xi++) {
			// If this is the locally owned part of the grid
			if (xi >= xs && xi < xs + xm) {
				// Get the distance from the surface
				double distance =
					(grid[xi] + grid[xi + 1]) / 2.0 - grid[surfacePos + 1];

				// Get the pointer to the beginning of the solution data for
				// this grid point
				gridPointSolution = solutionArray[xi];

				using HostUnmanaged = Kokkos::View<double*, Kokkos::HostSpace,
					Kokkos::MemoryUnmanaged>;
				auto hConcs = HostUnmanaged(gridPointSolution, dof);
				auto dConcs = Kokkos::View<double*>("Concentrations", dof);
				deep_copy(dConcs, hConcs);

				// Compute the helium density at this grid point
				double heDensity =
					psiNetwork->getTotalAtomConcentration(dConcs, specIdHe, 1);

				// Compute the radius of the bubble from the number of helium
				double nV = heDensity * (grid[xi + 1] - grid[xi]) / heVRatio;
				double latticeParam = network.getLatticeParameter();
				double tlcCubed = latticeParam * latticeParam * latticeParam;
				double radius = (sqrt(3.0) / 4) * latticeParam +
					cbrt((3.0 * tlcCubed * nV) / (8.0 * core::pi)) -
					cbrt((3.0 * tlcCubed) / (8.0 * core::pi));

				// Add randomness
				double prob = prefactor *
					(1.0 - (distance - radius) / distance) *
					std::min(1.0,
						exp(-(distance - depthParam) / (depthParam * 2.0)));
				double test = solverHandler.getRNG().GetRandomDouble();

				// If the bubble is too big or the probability is high enough
				if (prob > test || radius > distance) {
					burst = true;
					depthPositions1D.push_back(xi);
				}
			}
		}

		// If at least one grid point is bursting
		int localFlag = 1;
		if (burst) {
			// The event is happening
			localFlag = 0;
		}
		// All the processes should call post event
		int flag = -1;
		MPI_Allreduce(&localFlag, &flag, 1, MPI_INT, MPI_MIN, xolotlComm);
		fvalue[2] = flag;
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "postEventFunction1D")
/**
 * This is a method that moves the surface or burst bubbles
 */
PetscErrorCode
postEventFunction1D(TS ts, PetscInt nevents, PetscInt eventList[],
	PetscReal time, Vec solution, PetscBool, void*)
{
	perf::ScopedTimer myTimer(postEventFuncTimer);

	// Initial declaration
	PetscErrorCode ierr;
	double **solutionArray, *gridPointSolution;
	IdType xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Check if the surface has moved or a bubble burst
	if (nevents == 0) {
		PetscFunctionReturn(0);
	}

	// Check if both events happened
	if (nevents == 3)
		throw std::runtime_error(
			"\nxolotlSolver::Monitor1D: This is not supposed to "
			"happen, the surface cannot "
			"move in both directions at the same time!!");

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

	// Get the solver handler and local coordinates
	auto& solverHandler = PetscSolver::getSolverHandler();
	solverHandler.getLocalCoordinates(xs, xm, Mx, ys, ym, My, zs, zm, Mz);

	// Get the position of the surface
	auto surfacePos = solverHandler.getSurfacePosition();

	// Get the network
	auto& network = solverHandler.getNetwork();
	auto dof = network.getDOF();

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// Get the flux handler to know the flux amplitude.
	auto fluxHandler = solverHandler.getFluxHandler();
	double heliumFluxAmplitude = fluxHandler->getFluxAmplitude();

	// Get the delta time from the previous timestep to this timestep
	double previousTime = solverHandler.getPreviousTime();
	double dt = time - previousTime;

	// Take care of bursting
	using NetworkType = core::network::IPSIReactionNetwork;
	auto psiNetwork = dynamic_cast<NetworkType*>(&network);
	auto nBurst = std::vector<double>(3, 0.0);

	// Loop on each bursting depth
	for (auto i = 0; i < depthPositions1D.size(); i++) {
		// Get the pointer to the beginning of the solution data for this grid
		// point
		gridPointSolution = solutionArray[depthPositions1D[i]];

		// Get the distance from the surface
		auto xi = depthPositions1D[i];
		double distance =
			(grid[xi] + grid[xi + 1]) / 2.0 - grid[surfacePos + 1];
		double hxLeft = 0.0;
		if (xi < 1) {
			hxLeft = grid[xi + 1] - grid[xi];
		}
		else {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
		}

		// Write the bursting information
		std::ofstream outputFile;
		outputFile.open("bursting.txt", std::ios::app);
		outputFile << time << " " << distance << std::endl;
		outputFile.close();

		// Pinhole case
		psiNetwork->updateBurstingConcs(gridPointSolution, hxLeft, nBurst);
	}

	// Add up the local quantities
	auto globalBurst = std::vector<double>(3, 0.0);
	MPI_Allreduce(
		nBurst.data(), globalBurst.data(), 3, MPI_DOUBLE, MPI_SUM, xolotlComm);
	nHeliumBurst1D += globalBurst[0];
	nDeuteriumBurst1D += globalBurst[1];
	nTritiumBurst1D += globalBurst[2];

	// Now takes care of moving surface
	bool moving = false;
	bool movingUp = false;
	for (auto i = 0; i < nevents; i++) {
		if (eventList[i] < 2)
			moving = true;
		if (eventList[i] == 0)
			movingUp = true;
	}

	// Skip if nothing is moving
	if (!moving) {
		// Restore the solutionArray
		ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
		CHKERRQ(ierr);

		PetscFunctionReturn(0);
	}

	// Set the surface position
	auto xi = surfacePos + solverHandler.getLeftOffset();

	// Get the initial vacancy concentration
	double initialVConc = solverHandler.getInitialVConc();

	// The density of tungsten is 62.8 atoms/nm3, thus the threshold is
	double threshold = (62.8 - initialVConc) * (grid[xi] - grid[xi - 1]);

	if (movingUp) {
		int nGridPoints = 0;
		// Move the surface up until it is smaller than the next threshold
		while (nInterEvent1D > threshold) {
			// Move the surface higher
			surfacePos--;
			xi = surfacePos + solverHandler.getLeftOffset();
			nGridPoints++;
			// Update the number of interstitials
			nInterEvent1D -= threshold;
			// Update the thresold
			threshold = (62.8 - initialVConc) * (grid[xi] - grid[xi - 1]);
		}

		// Throw an exception if the position is negative
		if (surfacePos < 0) {
			PetscBool flagCheck;
			ierr =
				PetscOptionsHasName(NULL, NULL, "-check_collapse", &flagCheck);
			CHKERRQ(ierr);
			if (flagCheck) {
				// Write the convergence reason
				std::ofstream outputFile;
				outputFile.open("solverStatus.txt");
				outputFile << "overgrid" << std::endl;
				outputFile.close();
			}
			throw std::runtime_error(
				"\nxolotlSolver::Monitor1D: The surface is "
				"trying to go outside of the grid!!");
		}

		// Printing information about the extension of the material
		if (procId == 0) {
			std::cout << "Adding " << nGridPoints
					  << " points to the grid at time: " << time << " s."
					  << std::endl;
		}

		// Set it in the solver
		solverHandler.setSurfacePosition(surfacePos);

		// Initialize the vacancy concentration and the temperature on the new
		// grid points Get the single vacancy ID
		auto singleVacancyCluster = network.getSingleVacancy();
		auto vacancyIndex = core::network::IReactionNetwork::invalidIndex();
		if (singleVacancyCluster.getId() !=
			core::network::IReactionNetwork::invalidIndex())
			vacancyIndex = singleVacancyCluster.getId();
		// Get the surface temperature
		double temp = 0.0;
		if (xi >= xs && xi < xs + xm) {
			temp = solutionArray[xi][dof];
		}
		double surfTemp = 0.0;
		MPI_Allreduce(&temp, &surfTemp, 1, MPI_DOUBLE, MPI_SUM, xolotlComm);

		// Loop on the new grid points
		while (nGridPoints >= 0) {
			// Position of the newly created grid point
			xi = surfacePos + nGridPoints;

			// If xi is on this process
			if (xi >= xs && xi < xs + xm) {
				// Get the concentrations
				gridPointSolution = solutionArray[xi];

				// Set the new surface temperature
				gridPointSolution[dof] = surfTemp;

				if (vacancyIndex !=
						core::network::IReactionNetwork::invalidIndex() &&
					nGridPoints > 0) {
					// Initialize the vacancy concentration
					gridPointSolution[vacancyIndex] = initialVConc;
				}
			}

			// Decrease the number of grid points
			--nGridPoints;
		}
	}

	// Moving the surface back
	else {
		// Move it back as long as the number of interstitials in negative
		while (nInterEvent1D < 0.0) {
			// Compute the threshold to a deeper grid point
			threshold = (62.8 - initialVConc) * (grid[xi + 1] - grid[xi]);
			// Set all the concentrations to 0.0 at xi = surfacePos + 1
			// if xi is on this process
			if (xi >= xs && xi < xs + xm) {
				// Get the concentrations at xi = surfacePos + 1
				gridPointSolution = solutionArray[xi];
				// Loop on DOF
				for (auto i = 0; i < dof; i++) {
					gridPointSolution[i] = 0.0;
				}
			}

			// Move the surface deeper
			surfacePos++;
			xi = surfacePos + solverHandler.getLeftOffset();
			// Update the number of interstitials
			nInterEvent1D += threshold;
		}

		// Printing information about the extension of the material
		if (procId == 0) {
			std::cout << "Removing grid points to the grid at time: " << time
					  << " s." << std::endl;
		}

		// Set it in the solver
		solverHandler.setSurfacePosition(surfacePos);
	}

	// Set the new surface location in the surface advection handler
	auto advecHandler = solverHandler.getAdvectionHandler();
	advecHandler->setLocation(grid[surfacePos + 1] - grid[1]);

	// Set the new surface in the temperature handler
	auto tempHandler = solverHandler.getTemperatureHandler();
	tempHandler->updateSurfacePosition(surfacePos);

	// Get the flux handler to reinitialize it
	fluxHandler->initializeFluxHandler(
		solverHandler.getNetwork(), surfacePos, grid);

	// Write the updated surface position
	if (procId == 0) {
		std::ofstream outputFile;
		outputFile.open("surface.txt", std::ios::app);
		outputFile << time << " " << grid[surfacePos + 1] - grid[1]
				   << std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 * This operation sets up different monitors
 * depending on the options.
 *
 * @param ts The time stepper
 * @return A standard PETSc error code
 */
PetscErrorCode
setupPetsc1DMonitor(TS ts)
{
	PetscErrorCode ierr;

	auto handlerRegistry = perf::PerfHandlerRegistry::get();

	// Initialize the timers, including the one for this function.
	initTimer = handlerRegistry->getTimer("monitor1D:init");
	perf::ScopedTimer myTimer(initTimer);
	checkNegativeTimer = handlerRegistry->getTimer("monitor1D:checkNeg");
	tridynTimer = handlerRegistry->getTimer("monitor1D:tridyn");
	startStopTimer = handlerRegistry->getTimer("monitor1D:startStop");
	heRetentionTimer = handlerRegistry->getTimer("monitor1D:heRet");
	xeRetentionTimer = handlerRegistry->getTimer("monitor1D:xeRet");
	scatterTimer = handlerRegistry->getTimer("monitor1D:scatter");
	seriesTimer = handlerRegistry->getTimer("monitor1D:series");
	eventFuncTimer = handlerRegistry->getTimer("monitor1D:event");
	postEventFuncTimer = handlerRegistry->getTimer("monitor1D:postEvent");

	// Get the process ID
	auto xolotlComm = util::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get xolotlViz handler registry
	auto vizHandlerRegistry = viz::VizHandlerRegistry::get();

	// Flags to launch the monitors or not
	PetscBool flagNeg, flagCollapse, flag2DPlot, flag1DPlot, flagSeries,
		flagPerf, flagHeRetention, flagStatus, flagXeRetention, flagTRIDYN,
		flagAlloy, flagTemp, flagLargest;

	// Check the option -check_negative
	ierr = PetscOptionsHasName(NULL, NULL, "-check_negative", &flagNeg);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-check_negative) failed.");

	// Check the option -check_collapse
	ierr = PetscOptionsHasName(NULL, NULL, "-check_collapse", &flagCollapse);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-check_collapse) failed.");

	// Check the option -plot_perf
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_perf", &flagPerf);
	checkPetscError(
		ierr, "setupPetsc1DMonitor: PetscOptionsHasName (-plot_perf) failed.");

	// Check the option -plot_series
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_series", &flagSeries);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-plot_series) failed.");

	// Check the option -plot_1d
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_1d", &flag1DPlot);
	checkPetscError(
		ierr, "setupPetsc1DMonitor: PetscOptionsHasName (-plot_1d) failed.");

	// Check the option -helium_retention
	ierr =
		PetscOptionsHasName(NULL, NULL, "-helium_retention", &flagHeRetention);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-helium_retention) failed.");

	// Check the option -xenon_retention
	ierr =
		PetscOptionsHasName(NULL, NULL, "-xenon_retention", &flagXeRetention);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-xenon_retention) failed.");

	// Check the option -start_stop
	ierr = PetscOptionsHasName(NULL, NULL, "-start_stop", &flagStatus);
	checkPetscError(
		ierr, "setupPetsc1DMonitor: PetscOptionsHasName (-start_stop) failed.");

	// Check the option -tridyn
	ierr = PetscOptionsHasName(NULL, NULL, "-tridyn", &flagTRIDYN);
	checkPetscError(
		ierr, "setupPetsc1DMonitor: PetscOptionsHasName (-tridyn) failed.");

	// Check the option -alloy
	ierr = PetscOptionsHasName(NULL, NULL, "-alloy", &flagAlloy);
	checkPetscError(
		ierr, "setupPetsc1DMonitor: PetscOptionsHasName (-alloy) failed.");

	// Check the option -temp_profile
	ierr = PetscOptionsHasName(NULL, NULL, "-temp_profile", &flagTemp);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-temp_profile) failed.");

	// Check the option -largest_conc
	ierr = PetscOptionsHasName(NULL, NULL, "-largest_conc", &flagLargest);
	checkPetscError(ierr,
		"setupPetsc1DMonitor: PetscOptionsHasName (-largest_conc) failed.");

	// Get the solver handler
	auto& solverHandler = PetscSolver::getSolverHandler();

	// Get the network and its size
	auto& network = solverHandler.getNetwork();
	const auto networkSize = network.getNumClusters();
	// Get the number of species
	auto numSpecies = network.getSpeciesListSize();

	// Create data depending on the boundary conditions
	if (solverHandler.getLeftOffset() == 1) {
		nSurf1D = std::vector<double>(numSpecies, 0.0);
		previousSurfFlux1D = std::vector<double>(numSpecies, 0.0);
	}
	if (solverHandler.getRightOffset() == 1) {
		nBulk1D = std::vector<double>(numSpecies, 0.0);
		previousBulkFlux1D = std::vector<double>(numSpecies, 0.0);
	}

	// Determine if we have an existing restart file,
	// and if so, it it has had timesteps written to it.
	std::unique_ptr<io::XFile> networkFile;
	std::unique_ptr<io::XFile::TimestepGroup> lastTsGroup;
	std::string networkName = solverHandler.getNetworkName();
	bool hasConcentrations = false;
	if (not networkName.empty()) {
		networkFile = std::make_unique<io::XFile>(networkName);
		auto concGroup = networkFile->getGroup<io::XFile::ConcentrationGroup>();
		hasConcentrations = (concGroup and concGroup->hasTimesteps());
		if (hasConcentrations) {
			lastTsGroup = concGroup->getLastTimestepGroup();
		}
	}

	// Set the post step processing to stop the solver if the time step
	// collapses
	if (flagCollapse) {
		// Find the threshold
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-check_collapse", &timeStepThreshold, &flag);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: PetscOptionsGetReal (-check_collapse) "
			"failed.");
		if (!flag)
			timeStepThreshold = 1.0e-16;

		// Set the post step process that tells the solver when to stop if the
		// time step collapse
		ierr = TSSetPostStep(ts, checkTimeStep);
		checkPetscError(
			ierr, "setupPetsc1DMonitor: TSSetPostStep (checkTimeStep) failed.");
	}

	// Set the monitor to check the negative concentrations
	if (flagNeg) {
		// Find the stride to know how often we want to check
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-check_negative", &negThreshold1D, &flag);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: PetscOptionsGetReal (-check_negative) "
			"failed.");
		if (!flag)
			negThreshold1D = 1.0e-30;

		// checkNegative1D will be called at each timestep
		ierr = TSMonitorSet(ts, checkNegative1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (checkNegative1D) failed.");
	}

	// Set the monitor to save the status of the simulation in hdf5 file
	if (flagStatus) {
		// Find the stride to know how often the HDF5 file has to be written
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-start_stop", &hdf5Stride1D, &flag);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: PetscOptionsGetReal (-start_stop) failed.");
		if (!flag)
			hdf5Stride1D = 1.0;

		// Compute the correct hdf5Previous1D for a restart
		// Get the last time step written in the HDF5 file
		if (hasConcentrations) {
			assert(lastTsGroup);

			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			hdf5Previous1D = (PetscInt)(previousTime / hdf5Stride1D);
		}

		// Don't do anything if both files have the same name
		if (hdf5OutputName1D != solverHandler.getNetworkName()) {
			PetscInt Mx;
			PetscErrorCode ierr;

			// Get the da from ts
			DM da;
			ierr = TSGetDM(ts, &da);
			checkPetscError(ierr, "setupPetsc1DMonitor: TSGetDM failed.");

			// Get the size of the total grid
			ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE,
				PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
				PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
				PETSC_IGNORE, PETSC_IGNORE);
			checkPetscError(ierr, "setupPetsc1DMonitor: DMDAGetInfo failed.");

			// Get the solver handler
			auto& solverHandler = PetscSolver::getSolverHandler();

			// Get the physical grid
			auto grid = solverHandler.getXGrid();

			// Create and initialize a checkpoint file.
			// We do this in its own scope so that the file
			// is closed when the file object goes out of scope.
			// We want it to close before we (potentially) copy
			// the network from another file using a single-process
			// MPI communicator.
			{
				io::XFile checkpointFile(hdf5OutputName1D, grid, xolotlComm);
			}

			// Copy the network group from the given file (if it has one).
			// We open the files using a single-process MPI communicator
			// because it is faster for a single process to do the
			// copy with HDF5's H5Ocopy implementation than it is
			// when all processes call the copy function.
			// The checkpoint file must be closed before doing this.
			writeNetwork(xolotlComm, solverHandler.getNetworkName(),
				hdf5OutputName1D, network);
		}

		// startStop1D will be called at each timestep
		ierr = TSMonitorSet(ts, startStop1D, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc1DMonitor: TSMonitorSet (startStop1D) failed.");
	}

	// If the user wants the surface to be able to move or bursting
	if (solverHandler.moveSurface() || solverHandler.burstBubbles()) {
		// Surface
		if (solverHandler.moveSurface()) {
			using NetworkType = core::network::IPSIReactionNetwork;
			using AmountType = NetworkType::AmountType;
			auto psiNetwork = dynamic_cast<NetworkType*>(&network);
			// Get the number of species
			auto numSpecies = psiNetwork->getSpeciesListSize();
			auto specIdI = psiNetwork->getInterstitialSpeciesId();

			// Initialize the composition
			auto comp = std::vector<AmountType>(numSpecies, 0);

			// Loop on interstital clusters
			bool iClusterExists = true;
			AmountType iSize = 1;
			while (iClusterExists) {
				comp[specIdI()] = iSize;
				auto clusterId = psiNetwork->findClusterId(comp);
				// Check that the helium cluster is present in the network
				if (clusterId != NetworkType::invalidIndex()) {
					iClusterIds1D.push_back(clusterId);
					iSize++;
				}
				else
					iClusterExists = false;
			}

			// Get the interstitial information at the surface if concentrations
			// were stored
			if (hasConcentrations) {
				assert(lastTsGroup);

				// Get the interstitial quantity from the HDF5 file
				nInterEvent1D = lastTsGroup->readData1D("nInterstitial");
				// Get the previous I flux from the HDF5 file
				previousIEventFlux1D = lastTsGroup->readData1D("previousFluxI");
				// Get the previous time from the HDF5 file
				double previousTime = lastTsGroup->readPreviousTime();
				solverHandler.setPreviousTime(previousTime);
			}

			// Get the sputtering yield
			sputteringYield1D = solverHandler.getSputteringYield();

			// Master process
			if (procId == 0) {
				// Clear the file where the surface will be written
				std::ofstream outputFile;
				outputFile.open("surface.txt");
				outputFile << "#time height" << std::endl;
				outputFile.close();
			}
		}

		// Set directions and terminate flags for the surface event
		PetscInt direction[3];
		PetscBool terminate[3];
		direction[0] = 0, direction[1] = 0, direction[2] = 0;
		terminate[0] = PETSC_FALSE, terminate[1] = PETSC_FALSE,
		terminate[2] = PETSC_FALSE;
		// Set the TSEvent
		ierr = TSSetEventHandler(ts, 3, direction, terminate, eventFunction1D,
			postEventFunction1D, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSSetEventHandler (eventFunction1D) failed.");

		if (solverHandler.burstBubbles() && procId == 0) {
			// Uncomment to clear the file where the bursting info will be
			// written
			std::ofstream outputFile;
			outputFile.open("bursting.txt");
			outputFile << "#time depth" << std::endl;
			outputFile.close();
		}
	}

	// Set the monitor to save 1D plot of xenon distribution
	if (flag1DPlot) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a ScatterPlot
			scatterPlot1D = vizHandlerRegistry->getPlot(viz::PlotType::SCATTER);

			scatterPlot1D->setLogScale();

			// Create and set the label provider
			auto labelProvider = std::make_shared<viz::LabelProvider>();
			labelProvider->axis1Label = "Xenon Size";
			labelProvider->axis2Label = "Concentration";

			// Give it to the plot
			scatterPlot1D->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider =
				std::make_shared<viz::dataprovider::CvsXDataProvider>();

			// Give it to the plot
			scatterPlot1D->setDataProvider(dataProvider);
		}

		// monitorScatter1D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorScatter1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (monitorScatter1D) failed.");
	}

	// Set the monitor to save 1D plot of many concentrations
	if (flagSeries) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a ScatterPlot
			seriesPlot1D = vizHandlerRegistry->getPlot(viz::PlotType::SERIES);

			// set the log scale
			//			seriesPlot1D->setLogScale();

			// Create and set the label provider
			auto labelProvider = std::make_shared<viz::LabelProvider>();
			labelProvider->axis1Label = "x Position on the Grid";
			labelProvider->axis2Label = "Concentration";

			// Give it to the plot
			seriesPlot1D->setLabelProvider(labelProvider);

			// To plot a maximum of 18 clusters of the whole benchmark
			const auto loopSize = std::min(18, (int)networkSize);

			// Create a data provider for each cluster in the network
			for (auto i = 0; i < loopSize; i++) {
				// Create the data provider
				auto dataProvider =
					std::make_shared<viz::dataprovider::CvsXDataProvider>();

				// Give it to the plot
				seriesPlot1D->addDataProvider(dataProvider);
			}
		}

		// monitorSeries1D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorSeries1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (monitorSeries1D) failed.");
	}

	// Set the monitor to save performance plots (has to be in parallel)
	if (flagPerf) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a ScatterPlot
			perfPlot = vizHandlerRegistry->getPlot(viz::PlotType::SCATTER);

			// Create and set the label provider
			auto labelProvider = std::make_shared<viz::LabelProvider>();
			labelProvider->axis1Label = "Process ID";
			labelProvider->axis2Label = "Solver Time";

			// Give it to the plot
			perfPlot->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider =
				std::make_shared<viz::dataprovider::CvsXDataProvider>();

			// Give it to the plot
			perfPlot->setDataProvider(dataProvider);
		}

		// monitorPerf will be called at each timestep
		ierr = TSMonitorSet(ts, monitorPerf, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc1DMonitor: TSMonitorSet (monitorPerf) failed.");
	}

	// Set the monitor to compute the helium retention
	if (flagHeRetention) {
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

			// Get the names of the species in the network
			std::vector<std::string> names;
			for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
				names.push_back(network.getSpeciesName(id));
			}

			// If the surface is a free surface
			if (solverHandler.getLeftOffset() == 1) {
				// Loop on the names
				for (auto i = 0; i < names.size(); i++) {
					// Create the n attribute name
					std::ostringstream nName;
					nName << "n" << names[i] << "Surf";
					// Read quantity attribute
					nSurf1D[i] = lastTsGroup->readData1D(nName.str());

					// Create the previous flux attribute name
					std::ostringstream prevFluxName;
					prevFluxName << "previousFlux" << names[i] << "Surf";
					// Read the attribute
					previousSurfFlux1D[i] =
						lastTsGroup->readData1D(prevFluxName.str());
				}
			}

			// If the bottom is a free surface
			if (solverHandler.getRightOffset() == 1) {
				// Loop on the names
				for (auto i = 0; i < names.size(); i++) {
					// Create the n attribute name
					std::ostringstream nName;
					nName << "n" << names[i] << "Bulk";
					// Read quantity attribute
					nBulk1D[i] = lastTsGroup->readData1D(nName.str());

					// Create the previous flux attribute name
					std::ostringstream prevFluxName;
					prevFluxName << "previousFlux" << names[i] << "Bulk";
					// Read the attribute
					previousBulkFlux1D[i] =
						lastTsGroup->readData1D(prevFluxName.str());
				}
			}

			// Bursting
			if (solverHandler.burstBubbles()) {
				// Read about the impurity fluxes in from bursting
				nHeliumBurst1D = lastTsGroup->readData1D("nHeliumBurst");
				nDeuteriumBurst1D = lastTsGroup->readData1D("nDeuteriumBurst");
				nTritiumBurst1D = lastTsGroup->readData1D("nTritiumBurst");
			}
		}

		// computeFluence will be called at each timestep
		ierr = TSMonitorSet(ts, computeFluence, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc1DMonitor: TSMonitorSet (computeFluence) failed.");

		// computeHeliumRetention1D will be called at each timestep
		ierr = TSMonitorSet(ts, computeHeliumRetention1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (computeHeliumRetention1D) "
			"failed.");

		// Master process
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("retentionOut.txt");
			outputFile << "#fluence ";
			for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
				auto speciesName = network.getSpeciesName(id);
				outputFile << speciesName << "_content ";
			}
			if (solverHandler.getRightOffset() == 1) {
				for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
					auto speciesName = network.getSpeciesName(id);
					outputFile << speciesName << "_bulk ";
				}
			}
			if (solverHandler.getLeftOffset() == 1) {
				for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
					auto speciesName = network.getSpeciesName(id);
					outputFile << speciesName << "_surface ";
				}
			}
			outputFile << "Helium_burst Deuterium_burst Tritium_burst"
					   << std::endl;
			outputFile.close();
		}
	}

	// Set the monitor to compute the xenon retention
	if (flagXeRetention) {
		// Get the da from ts
		DM da;
		ierr = TSGetDM(ts, &da);
		checkPetscError(ierr, "setupPetsc1DMonitor: TSGetDM failed.");
		// Get the local boundaries
		PetscInt xm;
		ierr = DMDAGetCorners(da, NULL, NULL, NULL, &xm, NULL, NULL);
		checkPetscError(ierr, "setupPetsc1DMonitor: DMDAGetCorners failed.");
		// Create the local vectors on each process
		solverHandler.createLocalNE(xm);

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
			ierr, "setupPetsc1DMonitor: TSMonitorSet (computeFluence) failed.");

		// computeXenonRetention1D will be called at each timestep
		ierr = TSMonitorSet(ts, computeXenonRetention1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (computeXenonRetention1D) "
			"failed.");

		// Master process
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("retentionOut.txt");
			outputFile << "#time Xenon_content radius partial_radius "
						  "partial_bubble_conc partial_size"
					   << std::endl;
			outputFile.close();
		}
	}

	// Set the monitor to output data for TRIDYN
	if (flagTRIDYN) {
		// computeTRIDYN1D will be called at each timestep
		ierr = TSMonitorSet(ts, computeTRIDYN1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (computeTRIDYN1D) failed.");
	}

	// Set the monitor to output data for Alloy
	if (flagAlloy) {
		if (procId == 0) {
			// Create/open the output files
			std::fstream outputFile;
			outputFile.open("Alloy.dat", std::fstream::out);
			outputFile << "#time_step time ";
			for (auto id = core::network::SpeciesId(numSpecies); id; ++id) {
				auto speciesName = network.getSpeciesName(id);
				outputFile << speciesName << "_density " << speciesName
						   << "_diameter " << speciesName << "_partial_density "
						   << speciesName << "_partial_diameter ";
			}
			outputFile << std::endl;
			outputFile.close();
		}

		// computeAlloy1D will be called at each timestep
		ierr = TSMonitorSet(ts, computeAlloy1D, NULL, NULL);
		checkPetscError(
			ierr, "setupPetsc1DMonitor: TSMonitorSet (computeAlloy1D) failed.");
	}

	// Set the monitor to compute the temperature profile
	if (flagTemp) {
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("tempProf.txt");

			// Get the da from ts
			DM da;
			ierr = TSGetDM(ts, &da);
			checkPetscError(ierr, "setupPetsc1DMonitor: TSGetDM failed.");

			// Get the total size of the grid
			PetscInt Mx;
			ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE,
				PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
				PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
				PETSC_IGNORE, PETSC_IGNORE);
			checkPetscError(ierr, "setupPetsc1DMonitor: DMDAGetInfo failed.");

			// Get the physical grid
			auto grid = solverHandler.getXGrid();
			// Get the position of the surface
			auto surfacePos = solverHandler.getSurfacePosition();

			// Loop on the entire grid
			for (auto xi = surfacePos + solverHandler.getLeftOffset();
				 xi < Mx - solverHandler.getRightOffset(); xi++) {
				// Set x
				double x = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];
				outputFile << x << " ";
			}
			outputFile << std::endl;
			outputFile.close();
		}

		// computeCumulativeHelium1D will be called at each timestep
		ierr = TSMonitorSet(ts, profileTemperature1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (profileTemperature1D) failed.");
	}

	// Set the monitor to monitor the concentration of the largest cluster
	if (flagLargest) {
		// Look for the largest cluster
		auto& network = solverHandler.getNetwork();
		largestClusterId1D = network.getLargestClusterId();

		// Find the threshold
		PetscBool flag;
		ierr = PetscOptionsGetReal(
			NULL, NULL, "-largest_conc", &largestThreshold1D, &flag);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: PetscOptionsGetReal (-largest_conc) failed.");

		// monitorLargest1D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorLargest1D, NULL, NULL);
		checkPetscError(ierr,
			"setupPetsc1DMonitor: TSMonitorSet (monitorLargest1D) failed.");
	}

	// Set the monitor to simply change the previous time to the new time
	// monitorTime will be called at each timestep
	ierr = TSMonitorSet(ts, monitorTime, NULL, NULL);
	checkPetscError(
		ierr, "setupPetsc1DMonitor: TSMonitorSet (monitorTime) failed.");

	PetscFunctionReturn(0);
}

} /* end namespace monitor */
} /* end namespace solver */
} /* end namespace xolotl */
