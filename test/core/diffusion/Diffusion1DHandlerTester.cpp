#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <fstream>
#include <iostream>

#include <boost/test/unit_test.hpp>

#include <xolotl/core/diffusion/Diffusion1DHandler.h>
#include <xolotl/core/network/PSIReactionNetwork.h>
#include <xolotl/options/Options.h>
#include <xolotl/test/CommandLine.h>
#include <xolotl/util/MPIUtils.h>

using namespace std;
using namespace xolotl;
using namespace core;
using namespace diffusion;

using Kokkos::ScopeGuard;
BOOST_GLOBAL_FIXTURE(ScopeGuard);

/**
 * This suite is responsible for testing the Diffusion1DHandler.
 */
BOOST_AUTO_TEST_SUITE(Diffusion1DHandler_testSuite)

/**
 * Method checking the initialization of the off-diagonal part of the Jacobian,
 * and the compute diffusion methods.
 */
BOOST_AUTO_TEST_CASE(checkDiffusion)
{
	// Create the option to create a network
	xolotl::options::Options opts;
	// Create a good parameter file
	std::string parameterFile = "param.txt";
	std::ofstream paramFile(parameterFile);
	paramFile << "netParam=8 0 0 1 0" << std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	test::CommandLine<2> cl{{"fakeXolotlAppNameForTests", parameterFile}};
	util::mpiInit(cl.argc, cl.argv);
	opts.readParams(cl.argc, cl.argv);

	std::remove(parameterFile.c_str());

	// Create a grid
	std::vector<double> grid;
	std::vector<double> temperatures;
	for (int l = 0; l < 5; l++) {
		grid.push_back((double)l);
		temperatures.push_back(1000.0);
	}

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
	// Get its size
	const int dof = network.getDOF();

	// Create the diffusion handler
	Diffusion1DHandler diffusionHandler(opts.getMigrationThreshold());

	// Create a collection of advection handlers
	std::vector<advection::IAdvectionHandler*> advectionHandlers;

	// Create ofill
	network::IReactionNetwork::SparseFillMap ofill;

	// Initialize it
	diffusionHandler.initializeOFill(network, ofill);
	diffusionHandler.initializeDiffusionGrid(advectionHandlers, grid, 5, 0);

	// Test which cluster diffuses
	BOOST_REQUIRE_EQUAL(ofill[1][0], 1); // He_1
	BOOST_REQUIRE_EQUAL(ofill[3][0], 3); // He_2
	BOOST_REQUIRE_EQUAL(ofill[5][0], 5); // He_3
	BOOST_REQUIRE_EQUAL(ofill[7][0], 7); // He_4
	BOOST_REQUIRE_EQUAL(ofill[9][0], 9); // He_5
	BOOST_REQUIRE_EQUAL(ofill[11][0], 11); // He_6
	BOOST_REQUIRE_EQUAL(ofill[13][0], 13); // He_7
	BOOST_REQUIRE_EQUAL(ofill[0][0], 0); // V_1

	// Check the total number of diffusing clusters
	BOOST_REQUIRE_EQUAL(diffusionHandler.getNumberOfDiffusing(), 8);

	// The size parameter in the x direction
	double hx = 1.0;

	// The arrays of concentration
	double concentration[3 * dof];
	double newConcentration[3 * dof];

	// Initialize their values
	for (int i = 0; i < 3 * dof; i++) {
		concentration[i] = (double)i * i;
		newConcentration[i] = 0.0;
	}

	// Set the temperature to 1000K to initialize the diffusion coefficients
	network.setTemperatures(temperatures, grid);
	network.syncClusterDataOnHost();

	// Get pointers
	double* conc = &concentration[0];
	double* updatedConc = &newConcentration[0];

	// Get the offset for the grid point in the middle
	// Supposing the 3 grid points are laid-out as follow:
	// 0 | 1 | 2
	double* concOffset = conc + dof;
	double* updatedConcOffset = updatedConc + dof;

	// Fill the concVector with the pointer to the middle, left, and right grid
	// points
	double* concVector[3]{};
	concVector[0] = concOffset; // middle
	concVector[1] = conc; // left
	concVector[2] = conc + 2 * dof; // right

	// Compute the diffusion at this grid point
	diffusionHandler.computeDiffusion(
		network, concVector, updatedConcOffset, hx, hx, 0);

	// Check the new values of updatedConcOffset
	BOOST_REQUIRE_CLOSE(updatedConcOffset[1], 3.7081e+12, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[3], 1.8160e+12, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[5], 7.3065e+11, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[7], 9.6476e+11, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[9], 7.1800e+11, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[11], 1.7783e+10, 0.01);
	BOOST_REQUIRE_CLOSE(updatedConcOffset[13], 2.7860e+09, 0.01);
	BOOST_REQUIRE_CLOSE(
		updatedConcOffset[15], 0.0, 0.01); // He_8 does not diffuse
	BOOST_REQUIRE_CLOSE(updatedConcOffset[0], 2.9207e+08, 0.01);

	// Initialize the indices and values to set in the Jacobian
	int nDiff = diffusionHandler.getNumberOfDiffusing();
	IdType indices[nDiff];
	double val[3 * nDiff];
	// Get the pointer on them for the compute diffusion method
	IdType* indicesPointer = &indices[0];
	double* valPointer = &val[0];

	// Compute the partial derivatives for the diffusion a the grid point 1
	diffusionHandler.computePartialsForDiffusion(
		network, valPointer, indicesPointer, hx, hx, 0);

	// Check the values for the indices
	BOOST_REQUIRE_EQUAL(indices[0], 0);
	BOOST_REQUIRE_EQUAL(indices[1], 1);
	BOOST_REQUIRE_EQUAL(indices[2], 3);
	BOOST_REQUIRE_EQUAL(indices[3], 5);
	BOOST_REQUIRE_EQUAL(indices[4], 7);
	BOOST_REQUIRE_EQUAL(indices[5], 9);
	BOOST_REQUIRE_EQUAL(indices[6], 11);
	BOOST_REQUIRE_EQUAL(indices[7], 13);

	// Check some values
	BOOST_REQUIRE_CLOSE(val[1], 505312, 0.01);
	BOOST_REQUIRE_CLOSE(val[4], 6415444736, 0.01);
	BOOST_REQUIRE_CLOSE(val[5], 6415444736, 0.01);
	BOOST_REQUIRE_CLOSE(val[6], -6283827232, 0.01);
	BOOST_REQUIRE_CLOSE(val[9], -2528210084, 0.01);

	// Finalize MPI
	MPI_Finalize();
}

BOOST_AUTO_TEST_SUITE_END()
