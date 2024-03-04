#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <algorithm>
#include <fstream>
#include <iostream>

#include <boost/test/unit_test.hpp>

#include <xolotl/core/network/PSIReactionNetwork.h>
#include <xolotl/test/CommandLine.h>

using namespace std;
using namespace xolotl;
using namespace core;
using namespace network;

using Kokkos::ScopeGuard;
BOOST_GLOBAL_FIXTURE(ScopeGuard);

/**
 * This suite is responsible for testing the PSI network.
 */
BOOST_AUTO_TEST_SUITE(bursting_testSuite)

BOOST_AUTO_TEST_CASE(FullyRefined)
{
	// Create the option to create a network
	xolotl::options::Options opts;
	// Create a good parameter file
	std::string parameterFile = "param.txt";
	std::ofstream paramFile(parameterFile);
	paramFile << "netParam=8 0 0 2 2" << std::endl
			  << "process=reaction bursting" << std::endl
			  << "burstingFactor=1.0e8" << std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	test::CommandLine<2> cl{{"fakeXolotlAppNameForTests", parameterFile}};
	opts.readParams(cl.argc, cl.argv);

	std::remove(parameterFile.c_str());

	// Suppose we have a grid with 5 grid points and distance of
	// 0.1 nm between grid points
	std::vector<double> grid;
	for (int l = 0; l < 5; l++) {
		grid.push_back((double)l * 0.1);
	}

	using NetworkType = PSIReactionNetwork<PSIHeliumSpeciesList>;
	using Spec = NetworkType::Species;
	using Composition = NetworkType::Composition;

	// Get the boundaries from the options
	// Get the boundaries from the options
	NetworkType::AmountType maxV = opts.getMaxV();
	NetworkType::AmountType maxI = opts.getMaxI();
	NetworkType::AmountType maxHe = psi::getMaxHePerV(maxV, opts.getHeVRatio());
	NetworkType network({maxHe, maxV, maxI}, grid.size(), opts);

	network.syncClusterDataOnHost();
	network.getSubpaving().syncZones(plsm::onHost);

	BOOST_REQUIRE(!network.hasDeuterium());
	BOOST_REQUIRE(!network.hasTritium());
	auto deviceMemorySize = network.getDeviceMemorySize();
	BOOST_REQUIRE(deviceMemorySize > 370000);
	BOOST_REQUIRE(deviceMemorySize < 420000);

	BOOST_REQUIRE(network.getEnableStdReaction() == true);
	BOOST_REQUIRE(network.getEnableBursting() == true);

	BOOST_REQUIRE_EQUAL(network.getGridSize(), 5);

	// TODO: Test each value explicitly?
	typename NetworkType::Bounds bounds = network.getAllClusterBounds();
	BOOST_REQUIRE_EQUAL(bounds.size(), 35);
	typename NetworkType::PhaseSpace phaseSpace = network.getPhaseSpace();
	BOOST_REQUIRE_EQUAL(phaseSpace.size(), 3);

	BOOST_REQUIRE_EQUAL(network.getNumberOfSpecies(), 3);
	BOOST_REQUIRE_EQUAL(network.getNumberOfSpeciesNoI(), 2);

	// Check the single vacancy
	auto vacancy = network.getSingleVacancy();
	BOOST_REQUIRE_EQUAL(vacancy.getId(), 2);

	// Get the diagonal fill
	const auto dof = network.getDOF();
	NetworkType::SparseFillMap knownDFill;
	knownDFill[0] = {0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23,
		24, 26, 27, 29, 1, 4, 25, 28, 7, 22, 10, 19, 13, 16};
	knownDFill[1] = {
		1, 0, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 7, 25, 10, 22, 13, 19, 16};
	knownDFill[2] = {2, 0, 3, 1, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22,
		23, 25, 26, 28, 6, 9, 12, 15, 18, 21, 24, 27, 29};
	knownDFill[3] = {3, 0, 1, 2, 4, 7, 10, 13, 16, 19, 22, 6, 9, 12, 15, 18, 21,
		24, 27, 29, 30, 31, 32, 33, 34};
	knownDFill[4] = {4, 0, 5, 1, 6, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34};
	knownDFill[5] = {5, 0, 6, 2, 4, 7, 10, 13, 16, 19, 22, 8};
	knownDFill[6] = {6, 0, 1, 2, 5, 3, 4, 7, 10, 13, 16, 19, 22, 9};
	knownDFill[7] = {7, 0, 8, 1, 9, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
	knownDFill[8] = {8, 0, 9, 2, 7, 4, 5, 10, 13, 16, 19, 22, 11};
	knownDFill[9] = {9, 0, 1, 2, 8, 3, 7, 4, 6, 10, 13, 16, 19, 22, 12};
	knownDFill[10] = {10, 0, 11, 1, 12, 2, 3, 4, 7, 5, 6, 8, 9, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
	knownDFill[11] = {11, 0, 12, 2, 10, 4, 8, 5, 7, 13, 16, 19, 22, 14};
	knownDFill[12] = {12, 0, 1, 2, 11, 3, 10, 4, 9, 6, 7, 13, 16, 19, 22, 15};
	knownDFill[13] = {13, 0, 14, 1, 15, 2, 3, 4, 10, 5, 6, 7, 8, 9, 11, 12, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
	knownDFill[14] = {14, 0, 15, 2, 13, 4, 11, 5, 10, 7, 8, 16, 19, 22, 17};
	knownDFill[15] = {
		15, 0, 1, 2, 14, 3, 13, 4, 12, 6, 10, 7, 9, 16, 19, 22, 18};
	knownDFill[16] = {16, 0, 17, 1, 18, 2, 3, 4, 13, 5, 6, 7, 10, 8, 9, 11, 12,
		14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
	knownDFill[17] = {17, 0, 18, 2, 16, 4, 14, 5, 13, 7, 11, 8, 10, 19, 22, 20};
	knownDFill[18] = {
		18, 0, 1, 2, 17, 3, 16, 4, 15, 6, 13, 7, 12, 9, 10, 19, 22, 21};
	knownDFill[19] = {19, 0, 20, 1, 21, 2, 3, 4, 16, 5, 6, 7, 13, 8, 9, 10, 11,
		12, 14, 15, 17, 18, 22, 23, 24, 25, 26, 27};
	knownDFill[20] = {
		20, 0, 21, 2, 19, 4, 17, 5, 16, 7, 14, 8, 13, 10, 11, 22, 23};
	knownDFill[21] = {
		21, 0, 1, 2, 20, 3, 19, 4, 18, 6, 16, 7, 15, 9, 13, 10, 12, 22, 24};
	knownDFill[22] = {22, 0, 23, 1, 24, 2, 3, 4, 19, 5, 6, 7, 16, 8, 9, 10, 13,
		11, 12, 14, 15, 17, 18, 20, 21, 25};
	knownDFill[23] = {
		23, 0, 24, 2, 22, 4, 20, 5, 19, 7, 17, 8, 16, 10, 14, 11, 13, 26};
	knownDFill[24] = {
		24, 0, 1, 2, 23, 3, 22, 4, 21, 6, 19, 7, 18, 9, 16, 10, 15, 12, 13, 27};
	knownDFill[25] = {25, 0, 26, 1, 27, 2, 4, 22, 7, 19, 10, 16, 13};
	knownDFill[26] = {26, 0, 27, 2, 25, 4, 23, 5, 22, 7, 20, 8, 19, 10, 17, 11,
		16, 13, 14, 28};
	knownDFill[27] = {27, 0, 1, 2, 26, 4, 24, 6, 22, 7, 21, 9, 19, 10, 18, 12,
		16, 13, 15, 29};
	knownDFill[28] = {
		28, 0, 29, 2, 4, 25, 26, 7, 22, 23, 8, 10, 19, 20, 11, 13, 16, 17, 14};
	knownDFill[29] = {
		29, 0, 2, 28, 4, 27, 7, 24, 9, 22, 10, 21, 12, 19, 13, 18, 15, 16, 30};
	knownDFill[30] = {30, 4, 28, 29, 7, 25, 26, 27, 10, 22, 23, 24, 11, 12, 13,
		19, 20, 21, 14, 15, 16, 17, 18, 31};
	knownDFill[31] = {31, 4, 30, 7, 28, 29, 10, 25, 26, 27, 13, 22, 23, 24, 14,
		15, 16, 19, 20, 21, 17, 18, 32};
	knownDFill[32] = {32, 4, 31, 7, 30, 10, 28, 29, 13, 25, 26, 27, 16, 22, 23,
		24, 17, 18, 19, 20, 21, 33};
	knownDFill[33] = {33, 4, 32, 7, 31, 10, 30, 13, 28, 29, 16, 25, 26, 27, 19,
		22, 23, 24, 20, 21, 34};
	knownDFill[34] = {34, 4, 33, 7, 32, 10, 31, 13, 30, 16, 28, 29, 19, 25, 26,
		27, 22, 23, 24};

	NetworkType::SparseFillMap dfill;
	auto nPartials = network.getDiagonalFill(dfill);
	BOOST_REQUIRE_EQUAL(nPartials, 743);
	for (NetworkType::IndexType i = 0; i < dof; i++) {
		auto rowIter = dfill.find(i);
		if (rowIter != dfill.end()) {
			const auto& row = rowIter->second;
			BOOST_REQUIRE_EQUAL(row.size(), knownDFill[i].size());
		}
	}

	// Set temperatures
	std::vector<double> temperatures = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
	network.setTemperatures(temperatures);
	network.syncClusterDataOnHost();
	NetworkType::IndexType gridId = 1;

	// Create a concentration vector where every field is at 1.0
	std::vector<double> concentrations(dof + 1, 1.0);
	using HostUnmanaged =
		Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
	auto hConcs = HostUnmanaged(concentrations.data(), dof + 1);
	auto dConcs = Kokkos::View<double*>("Concentrations", dof + 1);
	deep_copy(dConcs, hConcs);

	// Define geometric positions
	auto surfacePos = grid[0];
	auto curXPos = (grid[gridId] + grid[gridId + 1]) / 2.0;
	auto prevXPos = (grid[gridId - 1] + grid[gridId]) / 2.0;
	auto curDepth = curXPos - surfacePos;
	auto curSpacing = curXPos - prevXPos;

	// Create a flux vector where every field is at 0.0
	std::vector<double> fluxes(dof + 1, 0.0);
	using HostUnmanaged =
		Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
	auto hFluxes = HostUnmanaged(fluxes.data(), dof + 1);
	auto dFluxes = Kokkos::View<double*>("Fluxes", dof + 1);
	deep_copy(dFluxes, hFluxes);

	std::vector<double> knownFluxes = {-6.59922e+12, -2.1875e+12, -3.05474e+11,
		-7.15605e+11, -9.27017e+11, -4.29437e+09, -6.78472e+11, -1.29538e+11,
		1.39068e+10, -6.58669e+11, 2.66124e+11, 2.14942e+10, -6.50437e+11,
		2.12395e+11, 3.17905e+10, -6.3929e+11, 2.99021e+11, 3.9628e+10,
		-6.30819e+11, 5.89806e+11, 3.98259e+10, -6.30606e+11, 5.85502e+11,
		3.98575e+10, -6.30572e+11, 6.01788e+11, 3.98891e+10, -6.30538e+11,
		4.52288e+11, -3.30415e+11, 1.66107e+11, 1.02851e+11, 8.18562e+10,
		7.82859e+10, 9.47596e+10, 0};

	// Check the fluxes computation
	network.computeAllFluxes(dConcs, dFluxes, gridId, curDepth, curSpacing);
	deep_copy(hFluxes, dFluxes);
	for (NetworkType::IndexType i = 0; i < dof + 1; i++) {
		BOOST_REQUIRE_CLOSE(fluxes[i], knownFluxes[i], 0.01);
	}

	std::vector<double> knownPartials = {-8.40783e+12, -2.30258e+10,
		-3.30546e+11, -2.90598e+11, -3.30546e+11, -2.90598e+11, -3.30546e+11,
		-2.90567e+11, -3.30546e+11, -2.90369e+11, -3.30546e+11, -2.82532e+11,
		-3.30546e+11, -2.72235e+11, -3.30546e+11, -2.64648e+11, -3.30546e+11,
		-2.46478e+11, -3.30546e+11, -3.30546e+11, 2.67577e+11, 9.09818e+10,
		5.573e+10, 7.91737e+10, 6.43611e+10, 2.81168e+10, 3.43955e+10,
		1.2623e+10, 6.74686e+10, 6.54704e+10, -2.96676e+12, 1.24349e+12,
		-2.67574e+11, -2.99909e+11, -2.99909e+11, -2.99909e+11, -2.99909e+11,
		-2.99909e+11, -2.99909e+11, -2.99909e+11, -2.99909e+11, -2.99909e+11,
		2.823e+10, 6.75752e+10, 2.31623e+10, 3.92905e+10, 4.66819e+10,
		2.9177e+10, 8.09113e+10, -6.37632e+11, 3.99456e+10, 3.31328e+11,
		-2.67574e+11, -3.52546e+10, 8.97668e+07, -1.82039e+10, 8.97668e+07,
		-7.59033e+09, 8.97668e+07, -1.02994e+10, 8.97668e+07, -7.84056e+09,
		8.97668e+07, -2.01138e+08, 8.97668e+07, -3.4839e+07, 8.97668e+07,
		-3.35622e+06, 8.97668e+07, 8.97668e+07, 0.589587, 7.13287e-10,
		1.64575e-21, 5.44315e-26, 2.53901e-30, 8.3304e-34, 5.3915e-39,
		1.30973e-42, 3.63678e-47, -7.17268e+11, -3.30546e+11, -2.99909e+11,
		6.973e+06, -3.85224e+10, -1.98028e+10, -8.23174e+09, -1.11472e+10,
		-8.47066e+09, -2.13572e+08, -3.39905e+07, 1.18556e+08, 1.18556e+08,
		1.18556e+08, 1.18556e+08, 1.18556e+08, 1.18556e+08, 1.18556e+08,
		1.18556e+08, 1.18556e+08, 1.18556e+08, 1.18556e+08, 1.18556e+08,
		1.18556e+08, 1.18556e+08, -1.7116e+12, 2.90598e+11, 2.55347e+11,
		2.99909e+11, 2.61387e+11, -3.52546e+10, -3.85224e+10, -7.43475e+10,
		-3.52518e+10, -3.85224e+10, -6.17954e+10, -3.52518e+10, -3.85224e+10,
		-6.64033e+10, -3.52518e+10, -3.85224e+10, -6.39522e+10, -3.52518e+10,
		-3.85224e+10, -5.4645e+10, -3.52518e+10, -3.85224e+10, -5.51305e+10,
		-3.52518e+10, -3.85224e+10, -5.573e+10, -3.52518e+10, -3.85224e+10,
		-3.52518e+10, -3.85224e+10, -3.85224e+10, -3.85224e+10, -3.85224e+10,
		-3.85224e+10, 0.168546, -3.70095e+11, 3.99475e+10, 3.30546e+11,
		3.52528e+10, 2.77661e+06, -1.8201e+10, -7.58729e+09, -1.02963e+10,
		-7.83737e+09, -1.97888e+08, -3.15332e+07, 0.000645754, -7.16996e+11,
		-3.30546e+11, -2.99909e+11, 1.74325e+06, 1.74325e+06, 3.85224e+10, 0,
		-1.98028e+10, -8.23174e+09, -1.11472e+10, -8.47066e+09, -2.13572e+08,
		-3.39905e+07, 8.5372e-13, -9.19042e+11, 2.90598e+11, 2.72397e+11,
		2.99909e+11, 2.80106e+11, -1.82039e+10, -1.98028e+10, 1.18575e+11,
		-1.8201e+10, -1.98028e+10, -3.67666e+10, -1.8201e+10, -1.98028e+10,
		-4.09504e+10, -1.8201e+10, -1.98028e+10, -3.79329e+10, -1.8201e+10,
		-1.98028e+10, -2.78409e+10, -1.8201e+10, -1.98028e+10, -2.79591e+10,
		-1.8201e+10, -1.98028e+10, -2.823e+10, -1.8201e+10, -1.98028e+10,
		-1.8201e+10, -1.98028e+10, -1.98028e+10, -1.98028e+10, -1.98028e+10,
		-3.70095e+11, 3.99475e+10, 3.30546e+11, 1.82022e+10, 2.92726e+06, 0,
		3.52518e+10, -7.58729e+09, -1.02963e+10, -7.83737e+09, -1.97888e+08,
		-3.15332e+07, 0.0122619, -7.16996e+11, -3.30546e+11, -2.99909e+11,
		1.74325e+06, 1.74325e+06, 1.98028e+10, 0, 0, 3.85224e+10, -8.23174e+09,
		-1.11472e+10, -8.47066e+09, -2.13572e+08, -3.39905e+07, 3.09165e-14,
		-4.42549e+11, 2.90598e+11, 2.83011e+11, 2.99909e+11, 2.91677e+11,
		-7.59033e+09, -8.23174e+09, 1.31121e+10, 3.81409e+10, -7.58729e+09,
		-8.23174e+09, -7.58729e+09, -8.23174e+09, -2.55803e+10, -7.58729e+09,
		-8.23174e+09, -2.22096e+10, -7.58729e+09, -8.23174e+09, -1.16336e+10,
		-7.58729e+09, -8.23174e+09, -1.15399e+10, -7.58729e+09, -8.23174e+09,
		-1.16223e+10, -7.58729e+09, -8.23174e+09, -7.58729e+09, -8.23174e+09,
		-8.23174e+09, -8.23174e+09, -3.70095e+11, 3.99475e+10, 3.30546e+11,
		7.58858e+09, 3.03294e+06, 0, 3.52518e+10, 1.8201e+10, 0, -1.02963e+10,
		-7.83737e+09, -1.97888e+08, -3.15332e+07, 0.01763, -7.16996e+11,
		-3.30546e+11, -2.99909e+11, 1.74325e+06, 1.74325e+06, 8.23174e+09, 0, 0,
		3.85224e+10, 1.98028e+10, 0, -1.11472e+10, -8.47066e+09, -2.13572e+08,
		-3.39905e+07, 6.37192e-07, -5.50394e+11, 2.90598e+11, 2.80302e+11,
		2.99909e+11, 2.88762e+11, -1.02994e+10, -1.11472e+10, -4.60539e+09,
		3.62176e+10, -1.02963e+10, -1.11472e+10, 6.13013e+10, -1.02963e+10,
		-1.11472e+10, -1.02963e+10, -1.11472e+10, -2.62623e+10, -1.02963e+10,
		-1.11472e+10, -1.55556e+10, -1.02963e+10, -1.11472e+10, -1.55021e+10,
		-1.02963e+10, -1.11472e+10, -1.56242e+10, -1.02963e+10, -1.11472e+10,
		-1.02963e+10, -1.11472e+10, -1.11472e+10, -3.70095e+11, 3.99475e+10,
		3.30546e+11, 1.02976e+10, 3.11707e+06, 0, 3.52518e+10, 7.58729e+09, 0,
		0, 1.8201e+10, -7.83737e+09, -1.97888e+08, -3.15332e+07, 3.74213,
		-7.16996e+11, -3.30546e+11, -2.99909e+11, 1.74325e+06, 1.74325e+06,
		1.11472e+10, 0, 0, 3.85224e+10, 8.23174e+09, 0, 0, 1.98028e+10,
		-8.47066e+09, -2.13572e+08, -3.39905e+07, 0.00019075, -4.40216e+11,
		2.90598e+11, 2.82761e+11, 2.99909e+11, 2.91439e+11, -7.84056e+09,
		-8.47066e+09, 2.42996e+09, 4.01199e+10, -7.83737e+09, -8.47066e+09,
		-1.16363e+09, 1.45596e+10, -7.83737e+09, -8.47066e+09, -7.83737e+09,
		-8.47066e+09, -7.83737e+09, -8.47066e+09, -1.18278e+10, -7.83737e+09,
		-8.47066e+09, -1.17238e+10, -7.83737e+09, -8.47066e+09, -1.18025e+10,
		-7.83737e+09, -8.47066e+09, -7.83737e+09, -8.47066e+09, -3.70095e+11,
		3.99475e+10, 3.30546e+11, 7.83882e+09, 3.18812e+06, 0, 3.52518e+10,
		1.02963e+10, 0, 0, 1.8201e+10, 7.58729e+09, 0, -1.97888e+08,
		-3.15332e+07, 0.269839, -7.16996e+11, -3.30546e+11, -2.99909e+11,
		1.74325e+06, 1.74325e+06, 8.47066e+09, 0, 0, 3.85224e+10, 1.11472e+10,
		0, 0, 1.98028e+10, 8.23174e+09, 0, -2.13572e+08, -3.39905e+07,
		9.67471e-05, -1.28411e+11, 2.90598e+11, 2.904e+11, 2.99909e+11,
		2.99696e+11, -2.01138e+08, -2.13572e+08, 9.32844e+09, 5.21457e+10,
		-1.97888e+08, -2.13572e+08, 1.31094e+10, 2.53948e+10, -1.97888e+08,
		-2.13572e+08, 3.16209e+10, -1.97888e+08, -2.13572e+08, -1.97888e+08,
		-2.13572e+08, -1.97888e+08, -2.13572e+08, -3.38777e+08, -1.97888e+08,
		-2.13572e+08, -2.96092e+08, -1.97888e+08, -2.13572e+08, -3.70095e+11,
		3.99475e+10, 3.30546e+11, 1.99395e+08, 3.25022e+06, 0, 3.52518e+10,
		7.83737e+09, 0, 0, 1.8201e+10, 1.02963e+10, 0, 0, 7.58729e+09,
		-3.15332e+07, 80.2838, -7.16996e+11, -3.30546e+11, -2.99909e+11,
		1.74325e+06, 1.74325e+06, 2.13572e+08, 0, 0, 3.85224e+10, 8.47066e+09,
		0, 0, 1.98028e+10, 1.11472e+10, 0, 0, 8.23174e+09, -3.39905e+07,
		0.000567812, -1.23349e+11, 2.90598e+11, 2.90567e+11, 2.99909e+11,
		2.99875e+11, -3.4839e+07, -3.39905e+07, -4.85626e+08, 5.43061e+10,
		-3.15332e+07, -3.39905e+07, 9.97379e+09, 2.62091e+10, -3.15332e+07,
		-3.39905e+07, 1.40405e+10, 1.00783e+10, -3.15332e+07, -3.39905e+07,
		-3.15332e+07, -3.39905e+07, -3.15332e+07, -3.39905e+07, -3.15332e+07,
		-3.39905e+07, 11.1828, -3.70095e+11, 3.99475e+10, 3.30546e+11,
		3.30957e+07, 3.30575e+06, 0, 3.52518e+10, 1.97888e+08, 0, 0, 1.8201e+10,
		7.83737e+09, 0, 0, 7.58729e+09, 1.02963e+10, 0, 1.61525, -7.16996e+11,
		-3.30546e+11, -2.99909e+11, 1.74325e+06, 1.74325e+06, 3.39905e+07, 0, 0,
		3.85224e+10, 2.13572e+08, 0, 0, 1.98028e+10, 8.47066e+09, 0, 0,
		8.23174e+09, 1.11472e+10, 0, 0.000428787, -1.23569e+11, 2.90598e+11,
		2.90598e+11, 2.99909e+11, 2.99909e+11, -3.35622e+06, -5.99321e+08,
		5.51307e+10, -3.89095e+08, 2.75449e+10, 1.05873e+10, 1.04071e+10,
		4.37128e+10, -3.70063e+11, 3.99475e+10, 3.30546e+11, 1.61296e+06,
		3.35622e+06, 0, 3.52518e+10, 3.15332e+07, 3.15332e+07, 0, 1.8201e+10,
		1.97888e+08, 0, 0, 7.58729e+09, 7.83737e+09, 0, 0, 1.02963e+10, 10.5114,
		-7.16962e+11, -3.30546e+11, -2.99909e+11, 1.74325e+06, 1.74325e+06, 0,
		3.85224e+10, 3.39905e+07, 3.39905e+07, 0, 1.98028e+10, 2.13572e+08, 0,
		0, 8.23174e+09, 8.47066e+09, 0, 0, 1.11472e+10, 0.000318953,
		-7.9267e+10, 3.30546e+11, 3.30546e+11, -1.74325e+06, 5.573e+10,
		5.573e+10, 3.52518e+10, 2.79591e+10, 2.79906e+10, 1.8201e+10,
		3.15332e+07, 1.16336e+10, 1.18315e+10, 7.58729e+09, 1.97888e+08,
		2.62836e+10, 2.62836e+10, 1.02963e+10, 7.83737e+09, -4.16839e+11,
		-3.30546e+11, 1.74325e+06, 1.74325e+06, 0, 3.85224e+10, 0, 1.98028e+10,
		3.39905e+07, 3.39905e+07, 0, 8.23174e+09, 2.13572e+08, 2.13572e+08, 0,
		1.11472e+10, 8.47066e+09, 0, 0.00691469, -7.78227e+10, 3.52518e+10,
		3.52518e+10, 3.85224e+10, 4.64311e+10, 2.823e+10, 1.8201e+10,
		1.98028e+10, 1.91272e+10, 1.16054e+10, 7.58729e+09, 8.23174e+09,
		3.15332e+07, 3.39905e+07, 2.58519e+10, 1.5967e+10, 1.02963e+10,
		1.11472e+10, 1.97888e+08, 2.13572e+08, 6.18652e+10, 7.83737e+09,
		8.47066e+09, 0.000243671, -6.66755e+10, 0, 3.85224e+10, 1.8201e+10,
		1.8201e+10, 1.98028e+10, 1.92096e+10, 1.16223e+10, 7.58729e+09,
		8.23174e+09, 3.69456e+10, 1.55677e+10, 1.02963e+10, 1.11472e+10,
		3.15332e+07, 3.39905e+07, 2.81359e+10, 1.22393e+10, 7.83737e+09,
		8.47066e+09, 1.97888e+08, 2.13572e+08, 0.0834275, -5.84438e+10, 0,
		3.85224e+10, 0, 1.98028e+10, 1.5819e+10, 7.58729e+09, 8.23174e+09,
		3.70677e+10, 1.56242e+10, 1.02963e+10, 1.11472e+10, 2.80318e+10,
		1.17893e+10, 7.83737e+09, 8.47066e+09, 3.15332e+07, 3.39905e+07,
		1.57001e+09, 1.97888e+08, 2.13572e+08, 0.0438128, -3.8641e+10, 0,
		3.85224e+10, 1.98028e+10, 1.98028e+10, 8.23174e+09, 8.23174e+09,
		2.14435e+10, 1.02963e+10, 1.11472e+10, 2.81106e+10, 1.18025e+10,
		7.83737e+09, 8.47066e+09, 7.50387e+08, 4.04451e+08, 1.97888e+08,
		2.13572e+08, 3.15332e+07, 3.39905e+07, 0.168546, -1.18556e+08,
		3.85224e+10, 3.85224e+10, 1.98028e+10, 1.98028e+10, 8.23174e+09,
		8.23174e+09, 1.11472e+10, 1.11472e+10, 1.6308e+10, 7.83737e+09,
		8.47066e+09, 7.07552e+08, 2.96092e+08, 1.97888e+08, 2.13572e+08,
		2.51275e+08, 3.15332e+07, 3.39905e+07};

	// Check the partials computation
	auto vals = Kokkos::View<double*>("solverPartials", nPartials);
	network.computeAllPartials(dConcs, vals, gridId, curDepth, curSpacing);
	auto hPartials = create_mirror_view(vals);
	deep_copy(hPartials, vals);
	int startingIdx = 0;
	for (NetworkType::IndexType i = 0; i < dof; i++) {
		auto rowIter = dfill.find(i);
		if (rowIter != dfill.end()) {
			const auto& row = rowIter->second;
			for (NetworkType::IndexType j = 0; j < row.size(); j++) {
				auto iter = find(row.begin(), row.end(), knownDFill[i][j]);
				auto index = std::distance(row.begin(), iter);
				BOOST_REQUIRE_CLOSE(hPartials[startingIdx + index],
					knownPartials[startingIdx + j], 0.01);
			}
			startingIdx += row.size();
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
