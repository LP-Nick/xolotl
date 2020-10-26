#include <cassert>
#include <fstream>
#include <limits>

#include <boost/program_options.hpp>

#include <xolotl/options/Options.h>
#include <xolotl/util/TokenizedLineReader.h>

namespace bpo = boost::program_options;

namespace xolotl
{
namespace options
{
Options::Options() :
	shouldRunFlag(true),
	exitCode(EXIT_SUCCESS),
	petscArg(""),
	networkFilename(""),
	tempHandlerName(""),
	tempParam{},
	tempProfileFilename(""),
	fluxFlag(false),
	fluxAmplitude(0.0),
	fluxTimeProfileFlag(false),
	perfHandlerName(""),
	vizHandlerName(""),
	materialName(""),
	initialVConcentration(0.0),
	voidPortion(50.0),
	dimensionNumber(1),
	useRegularGridFlag(true),
	useChebyshevGridFlag(false),
	readInGridFlag(false),
	gridFilename(""),
	gbList(""),
	groupingMin(std::numeric_limits<int>::max()),
	groupingWidthA(1),
	groupingWidthB(0),
	sputteringYield(0.0),
	useHDF5Flag(true),
	maxImpurity(8),
	maxD(0),
	maxT(0),
	maxV(20),
	maxI(6),
	nX(10),
	nY(0),
	nZ(0),
	xStepSize(0.5),
	yStepSize(0.0),
	zStepSize(0.0),
	leftBoundary(1),
	rightBoundary(1),
	bottomBoundary(1),
	topBoundary(1),
	frontBoundary(1),
	backBoundary(1),
	burstingDepth(10.0),
	burstingFactor(0.1),
	rngUseSeed(false),
	rngSeed(0),
	rngPrintSeed(false),
	zeta(0.73),
	density(10.162795276841),
	pulseTime(0.0),
	pulseProportion(0.0),
	latticeParameter(-1.0),
	impurityRadius(-1.0),
	biasFactor(1.15),
	hydrogenFactor(0.25),
	xenonDiffusivity(-1.0),
	fissionYield(0.25),
	heVRatio(4.0),
	migrationThreshold(std::numeric_limits<double>::infinity())
{
	return;
}

Options::~Options(void)
{
}

void
Options::readParams(int argc, char* argv[])
{
	// Check that a file name is given
	if (argc < 2) {
		std::cerr << "Options: parameter file name must not be empty"
				  << std::endl;
		shouldRunFlag = false;
		exitCode = EXIT_FAILURE;
		return;
	}

	// Check that the file exist
	std::ifstream ifs(argv[1]);
	if (!ifs) {
		std::cerr << "Options: unable to open parameter file: " << argv[1]
				  << std::endl;
		shouldRunFlag = false;
		exitCode = EXIT_FAILURE;
		return;
	}

	// The name of the parameter file
	std::string param_file;

	// Parse the command line options.
	bpo::options_description desc("Command line options");
	desc.add_options()("help", "show this help message")("parameterFile",
		bpo::value<std::string>(&param_file), "input file name");

	bpo::positional_options_description p;
	p.add("parameterFile", -1);

	bpo::variables_map opts;

	bpo::store(
		bpo::command_line_parser(argc, argv).options(desc).positional(p).run(),
		opts);
	bpo::notify(opts);

	// Declare a group of options that will be
	// allowed both on command line and in
	// config file
	bpo::options_description config("Parameters");
	config.add_options()("networkFile",
		bpo::value<std::string>(&networkFilename),
		"The network will be loaded from this HDF5 file.")("tempHandler",
		bpo::value<std::string>(&tempHandlerName)->default_value("constant"),
		"Temperature handler to use. (default = constant; available "
		"constant,gradient,heat,profile")("tempParam",
		bpo::value<std::string>(),
		"At most two parameters for temperature handler. Alternatives:"
		"constant -> temp; "
		"gradient -> surfaceTemp bulkTemp; "
		"heat -> heatFlux bulkTemp")("tempFile",
		bpo::value<std::string>(&tempProfileFilename),
		"A temperature profile is given by the specified file, "
		"then linear interpolation is used to fit the data."
		" NOTE: no need for tempParam here.")("flux",
		bpo::value<double>(&fluxAmplitude),
		"The value of the incoming flux in #/nm2/s. If the Fuel case is used "
		"it actually corresponds to the fission rate in #/nm3/s.")("fluxFile",
		bpo::value<std::string>(&fluxTimeProfileFilePath),
		"A time profile for the flux is given by the specified file, "
		"then linear interpolation is used to fit the data."
		"(NOTE: If a flux profile file is given, "
		"a constant flux should NOT be given)")("perfHandler",
		bpo::value<std::string>(&perfHandlerName)->default_value("os"),
		"Which set of performance handlers to use. (default = os, available "
		"dummy,os,papi).")("vizHandler",
		bpo::value<std::string>(&vizHandlerName)->default_value("dummy"),
		"Which set of handlers to use for the visualization. (default = dummy, "
		"available std,dummy).")("dimensions",
		bpo::value<int>(&dimensionNumber),
		"Number of dimensions for the simulation.")("material",
		bpo::value<std::string>(&materialName),
		"The material options are as follows: {W100, W110, W111, "
		"W211, Pulsed, Fuel, Fe, 800H}.")("initialV",
		bpo::value<double>(&initialVConcentration),
		"The value of the initial concentration of vacancies in the material.")(
		"zeta", bpo::value<double>(&zeta)->default_value(0.73),
		"The value of the electronic stopping power in the material (0.73 by "
		"default).")("voidPortion", bpo::value<double>(&voidPortion),
		"The value (in %) of the void portion at the start of the simulation.")(
		"regularGrid", bpo::value<std::string>(),
		"Will the grid be regularly spaced in the x direction? (available "
		"yes,no,cheby,<filename>)")("petscArgs",
		bpo::value<std::string>(&petscArg),
		"All the arguments that will be given to PETSc.")("process",
		bpo::value<std::string>(),
		"List of all the processes to use in the simulation (reaction, diff, "
		"advec, modifiedTM, movingSurface, bursting, attenuation, resolution, "
		"heterogeneous).")("grain", bpo::value<std::string>(&gbList),
		"This option allows the user to add GB in the X, Y, or Z directions. "
		"To do so, simply write the direction followed "
		"by the distance in nm, for instance: X 3.0 Z 2.5 Z 10.0 .")("grouping",
		bpo::value<std::string>(),
		"This option allows the use a grouping scheme starting at the cluster "
		"with 'min' size and with the given width.")("sputtering",
		bpo::value<double>(&sputteringYield),
		"This option allows the user to add a sputtering yield (atoms/ion).")(
		"netParam", bpo::value<std::string>(),
		"This option allows the user to define the boundaries of the network. "
		"To do so, simply write the values in order "
		"maxHe/Xe maxD maxT maxV maxI.")("grid", bpo::value<std::string>(),
		"This option allows the user to define the boundaries of the grid. "
		"To do so, simply write the values in order "
		"nX xStepSize nY yStepSize nZ zStepSize .")("radiusSize",
		bpo::value<std::string>(),
		"This option allows the user to set a minimum size for the computation "
		"for the average radii, in the same order as the netParam option "
		"(default is 0).")("boundary", bpo::value<std::string>(),
		"This option allows the user to choose the boundary conditions. "
		"The first one correspond to the left side (surface) "
		"and second one to the right (bulk), "
		"then two for Y and two for Z. "
		"0 means mirror or periodic, 1 means free surface.")("burstingDepth",
		bpo::value<double>(&burstingDepth),
		"This option allows the user to set a depth in nm "
		"for the bubble bursting.")("burstingFactor",
		bpo::value<double>(&burstingFactor),
		"This option allows the user to set the factor used in computing the "
		"likelihood of a bursting event.")("rng", bpo::value<std::string>(),
		"Allows user to specify seed used to initialize random number "
		"generator (default = determined from current time) and "
		"whether each process should print the seed value "
		"it uses (default = don't print)")("density",
		bpo::value<double>(&density),
		"This option allows the user to set a density in nm-3 "
		"for the number of xenon per volume in a bubble.")("pulse",
		bpo::value<std::string>(),
		"The total length of the pulse (in s) if the Pulsed material is used, "
		"and the proportion of it that is "
		"ON.")("lattice", bpo::value<double>(&latticeParameter),
		"This option allows the user to set the length of the lattice side in "
		"nm.")("impurityRadius", bpo::value<double>(&impurityRadius),
		"This option allows the user to set the radius of the main impurity "
		"(He or Xe) in nm.")("biasFactor", bpo::value<double>(&biasFactor),
		"This option allows the user to set the bias factor reflecting the "
		"fact that interstitial "
		"clusters have a larger surrounding strain field.")("hydrogenFactor",
		bpo::value<double>(&hydrogenFactor),
		"This option allows the user to set the factor between the size of He "
		"and H.")("xenonDiffusivity", bpo::value<double>(&xenonDiffusivity),
		"This option allows the user to set the diffusion coefficient for "
		"xenon in nm2 s-1.")("fissionYield", bpo::value<double>(&fissionYield),
		"This option allows the user to set the number of xenon created for "
		"each fission.")("heVRatio", bpo::value<double>(&heVRatio),
		"This option allows the user to set the number of He atoms allowed per "
		"V in a bubble.")("migrationThreshold",
		bpo::value<double>(&migrationThreshold),
		"This option allows the user to set a limit on the migration energy "
		"above which the diffusion will be ignored.")(
		"fluxDepthProfileFilePath",
		bpo::value<fs::path>(&fluxDepthProfileFilePath),
		"The path to the custom flux profile file; the default is an empty "
		"string that will use the default material associated flux handler.");

	bpo::options_description visible("Allowed options");
	visible.add(desc).add(config);

	if (opts.count("help")) {
		std::cout << visible << '\n';
		shouldRunFlag = false;
		exitCode = EXIT_FAILURE;
	}

	if (shouldRunFlag) {
		std::ifstream ifs(param_file);
		if (!ifs) {
			std::cerr << "Options: unable to open parameter file: "
					  << param_file << std::endl;
			exitCode = EXIT_FAILURE;
			return;
		}
		store(parse_config_file(ifs, config), opts);
		notify(opts);

		// Take care of the temperature
		if (opts.count("tempParam")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<double> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["tempParam"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();
			if (tokens.size() > 2) {
				std::cerr
					<< "Options: too many temperature parameters (expect 2)"
					<< std::endl;
				exitCode = EXIT_FAILURE;
				return;
			}
			for (std::size_t i = 0; i < tokens.size(); ++i) {
				tempParam[i] = tokens[i];
			}
		}

		if (opts.count("tempFile")) {
			// Check that the profile file exists
			std::ifstream inFile(tempProfileFilename);
			if (!inFile) {
				std::cerr << "\nOptions: could not open file containing "
							 "temperature profile data. "
							 "Aborting!\n"
						  << std::endl;
				shouldRunFlag = false;
				exitCode = EXIT_FAILURE;
			}
		}

		// Take care of the flux
		if (opts.count("flux")) {
			fluxFlag = true;
		}
		if (opts.count("fluxFile")) {
			// Check that the profile file exists
			std::ifstream inFile(fluxTimeProfileFilePath.c_str());
			if (!inFile) {
				std::cerr << "\nOptions: could not open file containing flux "
							 "profile data. "
							 "Aborting!\n"
						  << std::endl;
				shouldRunFlag = false;
				exitCode = EXIT_FAILURE;
			}
			else {
				// Set the flag to use a flux profile to true
				fluxTimeProfileFlag = true;
			}
		}

		// Take care of the performance handler
		if (opts.count("perfHandler")) {
			std::string perfHandlers[] = {"dummy", "std", "os", "papi"};
			if (std::find(begin(perfHandlers), end(perfHandlers),
					perfHandlerName) == end(perfHandlers)) {
				std::cerr << "\nOptions: could not understand the performance "
							 "handler type. "
							 "Aborting!\n"
						  << std::endl;
				shouldRunFlag = false;
				exitCode = EXIT_FAILURE;
			}
		}

		// Take care of the visualization handler
		if (opts.count("vizHandler")) {
			// Determine the type of handlers we are being asked to use
			if (!(vizHandlerName == "std" || vizHandlerName == "dummy")) {
				std::cerr << "\nOptions: unrecognized argument in the "
							 "visualization option handler."
							 "Aborting!\n"
						  << std::endl;
				shouldRunFlag = false;
				exitCode = EXIT_FAILURE;
			}
		}

		// Take care of the grid
		if (opts.count("regularGrid")) {
			auto arg = opts["regularGrid"].as<std::string>();
			// Determine the type of handlers we are being asked to use
			if (arg == "yes") {
				useRegularGridFlag = true;
			}
			else if (arg == "no") {
				useRegularGridFlag = false;
			}
			else if (arg == "cheby") {
				useChebyshevGridFlag = true;
			}
			else {
				// Read it as a file name
				gridFilename = arg;
				readInGridFlag = true;
			}
		}

		// Take care of the radius minimum size
		if (opts.count("radiusSize")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<int> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["radiusSize"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			// Set the values
			for (int i = 0; i < tokens.size(); i++) {
				radiusMinSizes.push_back(tokens[i]);
			}
		}

		// Take care of the processes
		if (opts.count("process")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<std::string> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["process"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			// Initialize the map of processes
			processMap["reaction"] = false;
			processMap["diff"] = false;
			processMap["advec"] = false;
			processMap["modifiedTM"] = false;
			processMap["movingSurface"] = false;
			processMap["bursting"] = false;
			processMap["attenuation"] = false;
			processMap["resolution"] = false;
			processMap["heterogeneous"] = false;

			// Loop on the tokens
			for (int i = 0; i < tokens.size(); ++i) {
				// Look for the key
				if (processMap.find(tokens[i]) == processMap.end()) {
					// Send an error
					std::cerr << "\nOptions: The process name is not known: "
							  << tokens[i] << std::endl
							  << "Aborting!\n"
							  << std::endl;
					shouldRunFlag = false;
					exitCode = EXIT_FAILURE;
				}
				else {
					// Switch the value to true in the map
					processMap[tokens[i]] = true;
				}
			}
		}

		// Take care of the gouping
		if (opts.count("grouping")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<int> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["grouping"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			// Set grouping minimum size
			groupingMin = tokens[0];
			// Set the grouping width in the first direction
			groupingWidthA = tokens[1];
			// Set the grouping width in the second direction
			if (tokens.size() > 2)
				groupingWidthB = tokens[2];
		}

		// Take care of the network parameters
		if (opts.count("netParam")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<std::string> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["netParam"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			// Set the flag to not use the HDF5 file
			useHDF5Flag = false;

			// Set the value for the impurities
			maxImpurity = strtol(tokens[0].c_str(), NULL, 10);

			// Check if we have other values
			if (tokens.size() > 1) {
				// Set the deuterium size
				maxD = strtol(tokens[1].c_str(), NULL, 10);
				// Set the tritium size
				maxT = strtol(tokens[2].c_str(), NULL, 10);
				// Set the vacancy size
				maxV = strtol(tokens[3].c_str(), NULL, 10);
				// Set the interstitial size
				maxI = strtol(tokens[4].c_str(), NULL, 10);
			}
		}

		// Take care of the grid
		if (opts.count("grid")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<std::string> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["grid"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			// Set the values for the for the depth
			nX = strtol(tokens[0].c_str(), NULL, 10);
			xStepSize = strtod(tokens[1].c_str(), NULL);

			// Check if we have other values
			if (tokens.size() > 2) {
				// Set the values for the for Y
				nY = strtol(tokens[2].c_str(), NULL, 10);
				yStepSize = strtod(tokens[3].c_str(), NULL);

				// Check if we have other values
				if (tokens.size() > 4) {
					// Set the values for the for Z
					nZ = strtol(tokens[4].c_str(), NULL, 10);
					zStepSize = strtod(tokens[5].c_str(), NULL);
				}
			}
		}

		// Take care of the boundary conditions
		if (opts.count("boundary")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<int> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["boundary"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			// Set the left boundary
			leftBoundary = tokens[0];
			// Set the right boundary
			rightBoundary = tokens[1];
			if (tokens.size() > 2)
				// Set the bottom boundary
				bottomBoundary = tokens[2];
			if (tokens.size() > 3)
				// Set the top boundary
				topBoundary = tokens[3];
			if (tokens.size() > 4)
				// Set the front boundary
				frontBoundary = tokens[4];
			if (tokens.size() > 5)
				// Set the back boundary
				backBoundary = tokens[5];
		}

		// Take care of the rng
		if (opts.count("rng")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<std::string> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["rng"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();
			try {
				size_t currIdx = 0;

				// Determine whether we should print the seed value.
				bool shouldPrintSeed = false;
				if (tokens[currIdx] == "print") {
					shouldPrintSeed = true;
					++currIdx;
				}
				rngPrintSeed = shouldPrintSeed;

				if (currIdx < tokens.size()) {
					// Convert arg to an integer.
					char* ep = NULL;
					auto useed = strtoul(tokens[currIdx].c_str(), &ep, 10);
					if (ep !=
						(tokens[currIdx].c_str() + tokens[currIdx].length())) {
						std::cerr
							<< "\nOptions: Invalid random number generator "
							   "seed, must be a non-negative integer."
							   "Aborting!\n"
							<< std::endl;
					}
					setRNGSeed(useed);
				}
			}
			catch (const std::invalid_argument& e) {
				std::cerr
					<< "\nOptions: unrecognized argument in setting the rng."
					   "Aborting!\n"
					<< std::endl;
				shouldRunFlag = false;
				exitCode = EXIT_FAILURE;
			}
		}

		// Take care of the flux pulse
		if (opts.count("pulse")) {
			// Build an input stream from the argument string.
			util::TokenizedLineReader<double> reader;
			auto argSS = std::make_shared<std::istringstream>(
				opts["pulse"].as<std::string>());
			reader.setInputStream(argSS);

			// Break the argument into tokens.
			auto tokens = reader.loadLine();

			pulseTime = tokens[0];
			pulseProportion = tokens[1];
		}
	}

	return;
}

} // end namespace options
} // end namespace xolotl
