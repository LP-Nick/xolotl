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

BOOST_AUTO_TEST_CASE(NE_1)
{
	if (getMPICommSize() > 1) {
		return;
	}
	// 0D, 9999 DOF ungrouped
	SystemTestCase{"benchmark_NE_1"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(NE_2)
{
	if (getMPICommSize() > 1) {
		return;
	}
	// 0D, 20079 DOF grouped
	SystemTestCase{"benchmark_NE_2"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(NE_3)
{
	if (getMPICommSize() > 400) {
		return;
		// 3D
	}
	SystemTestCase{"benchmark_NE_3"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(NE_4)
{
	if (getMPICommSize() > 3969 || getMPICommSize() < 16) {
		return;
		// 2D, longer
	}
	SystemTestCase{"benchmark_NE_4"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_1)
{
	if (getMPICommSize() > 10) {
		return;
	}
	SystemTestCase::copyFile("tridyn_benchmark_PSI_1.dat");
	// 1D + HeDTVI + grouping + heat
	SystemTestCase{"benchmark_PSI_1"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_2)
{
	if (getMPICommSize() > 20) {
		return;
	}
	SystemTestCase::copyFile("tridyn_benchmark_PSI_2.dat");
	// 1D + HeDVI + grouping + advection + modifiedTM + attenuation + surface +
	// reflective
	SystemTestCase{"benchmark_PSI_2"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_3)
{
	if (getMPICommSize() > 20) {
		return;
	}
	// 1D + HeVI + grouping + advection + modifiedTM + attenuation + surface +
	// reflective + reduced matrix method
	SystemTestCase{"benchmark_PSI_3"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_4)
{
	if (getMPICommSize() > 150 || getMPICommSize() < 8) {
		return;
	}
	// 1D + HeVI + advection + modifiedTM + attenuation + surface + reflective
	// longer
	SystemTestCase{"benchmark_PSI_4"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_5)
{
	if (getMPICommSize() > 150 || getMPICommSize() < 16) {
		return;
	}
	SystemTestCase::copyFile("tridyn_benchmark_PSI_5.dat");
	// 1D + HeDVI + advection + modifiedTM + attenuation + surface + reflective
	// + heat longer
	SystemTestCase{"benchmark_PSI_5"}.withTimer().run();
}

BOOST_AUTO_TEST_CASE(PSI_6)
{
	if (getMPICommSize() > 25 || getMPICommSize() < 4) {
		return;
	}
	// 1D + HeVI + pulsed flux + sink + I grouping + surface + reflective
	// bulk
	SystemTestCase{"benchmark_PSI_6"}.withTimer().run();
}

BOOST_AUTO_TEST_SUITE_END()
