#include "IReactionHandlerFactory.h"
#include "PSIReactionHandlerFactory.h"
#include "NEReactionHandlerFactory.h"
#include "AlloyReactionHandlerFactory.h"
#include "FeReactionHandlerFactory.h"
#include "UZrReactionHandlerFactory.h"

namespace xolotlFactory {

static std::shared_ptr<IReactionHandlerFactory> theReactionFactory;

std::shared_ptr<IReactionHandlerFactory> IReactionHandlerFactory::createNetworkFactory(
		const std::string &problemType) {
	// PSI case
	if (problemType == "W100" || problemType == "W110" || problemType == "W111"
			|| problemType == "W211" || problemType == "TRIDYN"
			|| problemType == "Pulsed")
		theReactionFactory = std::make_shared<PSIReactionHandlerFactory>();
	// NE case
	else if (problemType == "Fuel")
		theReactionFactory = std::make_shared<NEReactionHandlerFactory>();
	// Alloy case
	else if (problemType == "800H")
		theReactionFactory = std::make_shared<AlloyReactionHandlerFactory>();
	// Fe case
	else if (problemType == "Fe")
		theReactionFactory = std::make_shared<FeReactionHandlerFactory>();
	// UZr case
	else if (problemType == "UZr")
		theReactionFactory = std::make_shared<UZrReactionHandlerFactory>();
	// The type is not supported
	else {
		throw std::string(
				"\nThe problem type is not known: \"" + problemType + "\"");
	}

	return theReactionFactory;
}

} // end namespace xolotlFactory
