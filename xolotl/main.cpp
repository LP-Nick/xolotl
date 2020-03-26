#include <interface.h>

//! Main program
int main(int argc, char **argv) {

	// Initialize MPI
	MPI_Init(&argc, &argv);

	// Create an interface to control the solver
	XolotlInterface interface;

	// Initialize it
	interface.initializeXolotl(argc, argv);

	// Initialize the GB location
	interface.initGBLocation();

	// Run the solve
	interface.solveXolotl();

	// Check the convergence
	if (!interface.getConvergenceStatus()) {
		std::cout << "The largest concentration was too high." << std::endl;
	} else {
		std::cout << "Everything went fine." << std::endl;
	}
	// Finalize the run
	interface.finalizeXolotl();

	// Finalize MPI
	MPI_Finalize();

	return EXIT_SUCCESS;
}
