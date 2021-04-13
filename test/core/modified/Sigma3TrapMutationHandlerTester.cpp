#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <fstream>
#include <iostream>

#include <mpi.h>

#include <boost/test/unit_test.hpp>

#include <xolotl/core/advection/DummyAdvectionHandler.h>
#include <xolotl/core/advection/YGBAdvectionHandler.h>
#include <xolotl/core/modified/Sigma3TrapMutationHandler.h>
#include <xolotl/core/modified/W100TrapMutationHandler.h>
#include <xolotl/options/Options.h>

using namespace std;
using namespace xolotl;
using namespace core;
using namespace modified;

using Kokkos::ScopeGuard;
BOOST_GLOBAL_FIXTURE(ScopeGuard);

/**
 * This suite is responsible for testing the Sigma3TrapMutationHandler.
 */
BOOST_AUTO_TEST_SUITE(Sigma3TrapMutationHandler_testSuite)

/**
 * Method checking the initialization and the compute modified trap-mutation
 * methods.
 */
BOOST_AUTO_TEST_CASE(checkModifiedTrapMutation)
{
	// Create the option to create a network
	xolotl::options::Options opts;
	// Create a good parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 10 6" << std::endl
			  << "process=reaction" << std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char** argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array
	// Initialize MPI
	MPI_Init(&argc, &argv);
	opts.readParams(argc, argv);

	// Suppose we have a grid with 13 grid points and distance of
	// 0.1 nm between grid points
	IdType nGrid = 13;
	std::vector<double> grid;
	std::vector<double> temperatures;
	for (auto l = 0; l < nGrid; l++) {
		grid.push_back((double)l * 0.1);
		temperatures.push_back(1000.0);
	}
	// Set the surface position
	std::vector<IdType> surfacePos = {0, 0, 0, 0, 0};

	// Create the network
	using NetworkType =
		network::PSIReactionNetwork<network::PSIFullSpeciesList>;
	NetworkType::AmountType maxV = opts.getMaxV();
	NetworkType::AmountType maxI = opts.getMaxI();
	NetworkType::AmountType maxHe = opts.getMaxImpurity();
	NetworkType::AmountType maxD = opts.getMaxD();
	NetworkType::AmountType maxT = opts.getMaxT();
	NetworkType network({maxHe, maxD, maxT, maxV, maxI}, grid.size(), opts);
	network.syncClusterDataOnHost();
	network.getSubpaving().syncZones(plsm::onHost);
	// Get its size
	const auto dof = network.getDOF();

	// Create the modified trap-mutation handler
	W100TrapMutationHandler trapMutationHandler;

	// Create the advection handlers needed to initialize the trap mutation
	// handler
	std::vector<advection::IAdvectionHandler*> advectionHandlers;
	advectionHandlers.push_back(new advection::DummyAdvectionHandler());
	auto advecHandler = new advection::YGBAdvectionHandler();
	advecHandler->setLocation(1.0);
	advecHandler->setDimension(2);
	advectionHandlers.push_back(advecHandler);

	// Initialize it
	network::IReactionNetwork::SparseFillMap dfill;
	trapMutationHandler.initialize(network, dfill, 11, 5);
	trapMutationHandler.initializeIndex2D(
		surfacePos, network, advectionHandlers, grid, 11, 0, 5, 0.5, 0);

	// Check some values in dfill
	BOOST_REQUIRE_EQUAL(dfill[1][0], 71);
	BOOST_REQUIRE_EQUAL(dfill[27][0], 27);
	BOOST_REQUIRE_EQUAL(dfill[39][0], 38);

	// The arrays of concentration
	double concentration[nGrid * 5 * dof];
	double newConcentration[nGrid * 5 * dof];

	// Initialize their values
	for (auto i = 0; i < nGrid * 5 * dof; i++) {
		concentration[i] = (double)i * i;
		newConcentration[i] = 0.0;
	}

	// Get pointers
	double* conc = &concentration[0];
	double* updatedConc = &newConcentration[0];

	// Get the offset for the sixth grid point on the second row
	double* concOffset = conc + (nGrid * 1 + 5) * dof;
	double* updatedConcOffset = updatedConc + (nGrid * 1 + 5) * dof;

	// Set the temperature to compute the rates
	network.setTemperatures(temperatures);
	network.syncClusterDataOnHost();
	trapMutationHandler.updateTrapMutationRate(network.getLargestRate());

	// Compute the modified trap mutation at the sixth grid point
	trapMutationHandler.computeTrapMutation(
		network, concOffset, updatedConcOffset, 5, 1);

	// Check the new values of updatedConcOffset
	BOOST_REQUIRE_CLOSE(updatedConcOffset[0], 9.5324e+21, 0.01); // Create I
	BOOST_REQUIRE_CLOSE(updatedConcOffset[49], -2.3426e+21, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[50], 2.3426e+21, 0.01);

	// Get the offset for the ninth grid point on the fourth row
	concOffset = conc + (nGrid * 3 + 8) * dof;
	updatedConcOffset = updatedConc + (nGrid * 3 + 8) * dof;

	// Compute the modified trap mutation at the ninth grid point
	trapMutationHandler.computeTrapMutation(
		network, concOffset, updatedConcOffset, 8, 3);

	// Check the new values of updatedConcOffset
	BOOST_REQUIRE_CLOSE(updatedConcOffset[0], 6.23056e+22, 0.01); // Create I
	BOOST_REQUIRE_CLOSE(updatedConcOffset[49], -1.54727e+22, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[50], 1.54727e+22, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[60], -1.5542e+22, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[61], 1.5542e+22, 0.01);

	// Initialize the indices and values to set in the Jacobian
	IdType indices[3 * maxHe];
	double val[3 * maxHe];
	// Get the pointer on them for the compute modified trap-mutation method
	IdType* indicesPointer = &indices[0];
	double* valPointer = &val[0];

	// Compute the partial derivatives for the modified trap-mutation at the
	// grid point 8
	auto nMutating = trapMutationHandler.computePartialsForTrapMutation(
		network, concOffset, valPointer, indicesPointer, 8, 3);

	// Check the values for the indices
	BOOST_REQUIRE_EQUAL(nMutating, 4);
	BOOST_REQUIRE_EQUAL(indices[0], 49); // He4
	BOOST_REQUIRE_EQUAL(indices[1], 50); // He4V
	BOOST_REQUIRE_EQUAL(indices[2], 0); // I
	BOOST_REQUIRE_EQUAL(indices[3], 60); // He5
	BOOST_REQUIRE_EQUAL(indices[4], 61); // He5V
	BOOST_REQUIRE_EQUAL(indices[5], 0); // I

	// Check values
	BOOST_REQUIRE_CLOSE(val[0], -6.34804e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[1], 6.34804e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[2], 6.34804e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[3], -6.34804e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[4], 6.34804e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[5], 6.34804e+14, 0.01);

	// Change the temperature of the network
	for (auto l = 0; l < nGrid; l++) {
		temperatures[l] = 500.0;
	}
	network.setTemperatures(temperatures);
	network.syncClusterDataOnHost();

	// Update the bursting rate
	trapMutationHandler.updateTrapMutationRate(network.getLargestRate());

	// Compute the partial derivatives for the bursting a the grid point 8
	nMutating = trapMutationHandler.computePartialsForTrapMutation(
		network, concOffset, valPointer, indicesPointer, 8, 3);

	// Check values
	BOOST_REQUIRE_EQUAL(nMutating, 4);
	BOOST_REQUIRE_CLOSE(val[0], -5.53624e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[1], 5.53624e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[2], 5.53624e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[3], -5.53624e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[4], 5.53624e+14, 0.01);
	BOOST_REQUIRE_CLOSE(val[5], 5.53624e+14, 0.01);

	// Remove the created file
	std::string tempFile = "param.txt";
	std::remove(tempFile.c_str());

	// Finalize MPI
	MPI_Finalize();

	return;
}

BOOST_AUTO_TEST_SUITE_END()
