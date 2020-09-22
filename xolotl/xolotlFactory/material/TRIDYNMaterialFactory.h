#ifndef TRIDYNMATERIALHANDLERFACTORY_H
#define TRIDYNMATERIALHANDLERFACTORY_H

#include <memory>
#include <MaterialFactory.h>
#include <TRIDYNFitFluxHandler.h>
#include <W100AdvectionHandler.h>
#include <W100TrapMutationHandler.h>

namespace xolotlFactory {

/**
 * Subclass of MaterialFactory for a (100) oriented tungsten material with
 * a TRIDYN input file.
 */
class TRIDYNMaterialFactory: public MaterialFactory {
public:

	/**
	 * The constructor creates the handlers.
	 *
	 * @param dim The number of dimensions for the problem
	 */
	TRIDYNMaterialFactory(const xolotlCore::Options &options) :
			MaterialFactory(options) {
		theFluxHandler = std::make_shared<xolotlCore::TRIDYNFitFluxHandler>();
		theAdvectionHandler.push_back(
				std::make_shared<xolotlCore::W100AdvectionHandler>());
		theTrapMutationHandler = std::make_shared<
				xolotlCore::W100TrapMutationHandler>();
		theReSolutionHandler = std::make_shared<
				xolotlCore::DummyReSolutionHandler>();
		theNucleationHandler = std::make_shared<
				xolotlCore::DummyNucleationHandler>();

		return;
	}

	/**
	 * The destructor
	 */
	~TRIDYNMaterialFactory() {
	}

	/**
	 * Initialize the material conditions with the different given options.
	 *
	 * @param options The Xolotl options.
	 */
	void initializeMaterial(const xolotlCore::Options &options) {
		// First call the original one
		MaterialFactory::initializeMaterial(options);

		// Get the number of dimensions
		int dim = options.getDimensionNumber();

		// Set-up the grain boundaries from the options
		std::string gbString = options.getGbString();
		// Build an input stream from the GB string.
		xolotlCore::TokenizedLineReader<std::string> reader;
		auto argSS = std::make_shared<std::istringstream>(gbString);
		reader.setInputStream(argSS);
		// Break the string into tokens.
		auto tokens = reader.loadLine();
		// Loop on them
		for (int i = 0; i < tokens.size(); i++) {
			// Switch on the type of grain boundaries
			if (tokens[i] == "X") {
				auto GBAdvecHandler = std::make_shared<
						xolotlCore::XGBAdvectionHandler>();
				GBAdvecHandler->setLocation(
						strtod(tokens[i + 1].c_str(), NULL));
				GBAdvecHandler->setDimension(dim);
				theAdvectionHandler.push_back(GBAdvecHandler);
			} else if (tokens[i] == "Y") {
				if (dim < 2)
					// A Y grain boundary cannot be used in 1D.
					throw std::string(
							"\nA Y grain boundary CANNOT be used in 1D. Switch to 2D or 3D or remove it.");

				auto GBAdvecHandler = std::make_shared<
						xolotlCore::YGBAdvectionHandler>();
				GBAdvecHandler->setLocation(
						strtod(tokens[i + 1].c_str(), NULL));
				GBAdvecHandler->setDimension(dim);
				theAdvectionHandler.push_back(GBAdvecHandler);
			} else if (tokens[i] == "Z") {
				if (dim < 3)
					// A Z grain boundary cannot be used in 1D/2D.
					throw std::string(
							"\nA Z grain boundary CANNOT be used in 1D/2D. Switch to 3D or remove it.");

				auto GBAdvecHandler = std::make_shared<
						xolotlCore::ZGBAdvectionHandler>();
				GBAdvecHandler->setLocation(
						strtod(tokens[i + 1].c_str(), NULL));
				GBAdvecHandler->setDimension(dim);
				theAdvectionHandler.push_back(GBAdvecHandler);
			} else {
				// Wrong GB type
				throw std::string(
						"\nThe type of grain boundary is not known: \""
								+ tokens[i] + "\"");
			}

			i++;
		}

		return;
	}
};

} // end namespace xolotlFactory

#endif // TRIDYNMATERIALHANDLERFACTORY_H
