#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <boost/test/unit_test.hpp>

#include <xolotl/test/KokkosFixture.h>

BOOST_GLOBAL_FIXTURE(KokkosFixture);

#include <xolotl/test/MPITestUtils.h>
#include <xolotl/test/SystemTestCase.h>
using xolotl::test::getMPICommSize;
using xolotl::test::SystemTestCase;

BOOST_GLOBAL_FIXTURE(MPIFixture);

BOOST_AUTO_TEST_SUITE(Benchmark)

BOOST_AUTO_TEST_CASE(AZr_1)
{
	if (getMPICommSize() > 1) {
		return;
	}
	// 0D, 1000 in each direction, grouped
	SystemTestCase{"benchmark_AZr_1", "AlphaZr.dat"}.withTimer().run();
}

// BOOST_AUTO_TEST_CASE(NE_1)
//{
//	if (getMPICommSize() > 1) {
//		return;
//	}
// 0D, 10000 DOF ungrouped
//	SystemTestCase{"benchmark_NE_1"}.withTimer().run();
//}
//
// BOOST_AUTO_TEST_CASE(NE_2)
//{
//	if (getMPICommSize() > 1) {
//		return;
//	}
// 0D, 2010 DOF grouped
//	SystemTestCase{"benchmark_NE_2"}.withTimer().run();
//}
//
// BOOST_AUTO_TEST_CASE(NE_3)
//{
//	if (getMPICommSize() > 3969) {
//		return;
//	}
// 3D
//	SystemTestCase{"benchmark_NE_3"}.withTimer().run();
//}
//
// BOOST_AUTO_TEST_CASE(NE_4)
//{
//	if (getMPICommSize() > 15876 || getMPICommSize() < 16) {
//		return;
//	}
// 2D, longer
//	SystemTestCase{"benchmark_NE_4"}.withTimer().run();
//}
//
// BOOST_AUTO_TEST_CASE(NE_5)
//{
//	if (getMPICommSize() > 1) {
//		return;
//	}
// 0D, grouped, re-solution
//	SystemTestCase{"benchmark_NE_5"}.tolerance(5.0e-9).withTimer().run();
//}

BOOST_AUTO_TEST_CASE(PSI_1)
{
	if (getMPICommSize() > 100) {
		return;
	}
	// 1D + HeV + 4e25 flux W100
	SystemTestCase{"benchmark_PSI_1"}.tolerance(1.0e-5).withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_2)
{
	if (getMPICommSize() > 1) {
		return;
	}
	// 1D + + HeV + 5e27 flux W100 + bursting
	SystemTestCase{"benchmark_PSI_2"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_3)
{
	if (getMPICommSize() != 32) {
		return;
	}
	// 1D + HeV + case g + bursting
	SystemTestCase{"benchmark_PSI_3"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_4)
{
	if (getMPICommSize() != 32) {
		return;
	}
	SystemTestCase::copyFile("tridyn_benchmark_PSI_4.dat");
	// 1D + ITER_He
	SystemTestCase{"benchmark_PSI_4"}.tolerance(5.0e-10).withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_5)
{
	if (getMPICommSize() != 32) {
		return;
	}
	SystemTestCase::copyFile("tridyn_benchmark_PSI_5.dat");
	// 1D + ITER_BPO
	SystemTestCase{"benchmark_PSI_5"}.tolerance(5.0e-10).withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_7)
{
	if (getMPICommSize() < 4 || getMPICommSize() > 25) {
		return;
	}
	// 1D + pulsed
	SystemTestCase{"benchmark_PSI_7"}.tolerance(1.0e-8).withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_8)
{
	if (getMPICommSize() > 20) {
		return;
	}
	SystemTestCase::copyFile("flux_benchmark_PSI_8.dat");
	SystemTestCase::copyFile("tridyn_benchmark_PSI_8.dat");
	// 1D + PISCES + varying flux
	SystemTestCase{"benchmark_PSI_8"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_9)
{
	if (getMPICommSize() > 20) {
		return;
	}
	SystemTestCase::copyFile("temp_benchmark_PSI_9.dat");
	SystemTestCase::copyFile("tridyn_benchmark_PSI_9.dat");
	// 1D + PISCES + varying temperature
	SystemTestCase{"benchmark_PSI_9"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_10)
{
	if (getMPICommSize() > 20) {
		return;
	}
	// 1D + reduced jacobian
	SystemTestCase{"benchmark_PSI_10"}.withTimer().run();
}

BOOST_AUTO_TEST_SUITE_END()
