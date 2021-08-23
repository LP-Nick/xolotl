#include <fstream>
#include <sstream>

#include <xolotl/util/MPIUtils.h>

namespace xolotl
{
namespace util
{
// The MPI communicator
static MPI_Comm xolotlMPIComm = MPI_COMM_WORLD;

void
setMPIComm(MPI_Comm comm)
{
	xolotlMPIComm = comm;

	return;
}

MPI_Comm
getMPIComm(void)
{
	return xolotlMPIComm;
}

void
mpiInit(int& argc, const char* argv[])
{
	auto ncargv = const_cast<char**>(argv);
	MPI_Init(&argc, &ncargv);
}

} // namespace util
} // namespace xolotl
