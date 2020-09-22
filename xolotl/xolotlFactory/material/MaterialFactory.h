#ifndef MATERIALHANDLERFACTORY_H
#define MATERIALHANDLERFACTORY_H

#include <memory>
#include <IMaterialFactory.h>
#include <DummyDiffusionHandler.h>
#include <DummyAdvectionHandler.h>
#include <DummyTrapMutationHandler.h>
#include <DummyReSolutionHandler.h>
#include <DummyNucleationHandler.h>
#include <TokenizedLineReader.h>
#include <XGBAdvectionHandler.h>
#include <YGBAdvectionHandler.h>
#include <ZGBAdvectionHandler.h>
#include <Diffusion1DHandler.h>
#include <Diffusion2DHandler.h>
#include <Diffusion3DHandler.h>

namespace xolotlFactory {

/**
 * Realizes the IMaterialFactory interface. Handles the flux, the advection, modified
 * trap mutation, and for a specific material.
 */
class MaterialFactory: public IMaterialFactory {
protected:

	//! The flux handler
	std::shared_ptr<xolotlCore::IFluxHandler> theFluxHandler;

	//! The advection handler
	std::vector<std::shared_ptr<xolotlCore::IAdvectionHandler> > theAdvectionHandler;

	//! The diffusion handler
	std::shared_ptr<xolotlCore::IDiffusionHandler> theDiffusionHandler;

	//! The modified trap-mutation handler
	std::shared_ptr<xolotlCore::ITrapMutationHandler> theTrapMutationHandler;

	//! The re-solution handler
	std::shared_ptr<xolotlCore::IReSolutionHandler> theReSolutionHandler;

	//! The heterogeneous nucleation handler
	std::shared_ptr<xolotlCore::IHeterogeneousNucleationHandler> theNucleationHandler;

public:

	/**
	 * The constructor creates the handlers.
	 */
	MaterialFactory(const xolotlCore::Options &options) {
		// Get the dimension and migration energy threshold
		int dim = options.getDimensionNumber();
		double threshold = options.getMigrationThreshold();
		// Switch on the dimension for the diffusion handler
		switch (dim) {
		case 0:
			theDiffusionHandler = std::make_shared<
					xolotlCore::DummyDiffusionHandler>(threshold);
			break;
		case 1:
			theDiffusionHandler = std::make_shared<
					xolotlCore::Diffusion1DHandler>(threshold);
			break;
		case 2:
			theDiffusionHandler = std::make_shared<
					xolotlCore::Diffusion2DHandler>(threshold);
			break;
		case 3:
			theDiffusionHandler = std::make_shared<
					xolotlCore::Diffusion3DHandler>(threshold);
			break;
		default:
			// The asked dimension is not good (e.g. -1, 4)
			throw std::string(
					"\nxolotlFactory: Bad dimension for the material factory.");
		}
	}

	/**
	 * The destructor
	 */
	~MaterialFactory() {
	}

	/**
	 * Initialize the material conditions with the different given options.
	 *
	 * @param options The Xolotl options.
	 */
	virtual void initializeMaterial(const xolotlCore::Options &options) {
		// Wrong if both he flux and time profile options are used
		if (options.useFluxAmplitude() && options.useFluxTimeProfile()) {
			// A constant flux value AND a time profile cannot both be given.
			throw std::string(
					"\nA constant flux value AND a time profile cannot both be given.");
		} else if (options.useFluxAmplitude()) {
			// Set the constant value of the flux
			theFluxHandler->setFluxAmplitude(options.getFluxAmplitude());
		} else if (options.useFluxTimeProfile()) {
			// Initialize the time profile
			theFluxHandler->initializeTimeProfile(options.getFluxProfileName());
		}

		// Get the process map
		auto map = options.getProcesses();
		// Set dummy handlers when needed
		if (!map["diff"])
			theDiffusionHandler = std::make_shared<
					xolotlCore::DummyDiffusionHandler>(
					options.getMigrationThreshold());
		if (!map["advec"]) {
			// Clear the advection handler
			theAdvectionHandler.clear();
			// To replace it by a dummy one
			theAdvectionHandler.push_back(
					std::make_shared<xolotlCore::DummyAdvectionHandler>());
		}
		if (!map["modifiedTM"])
			theTrapMutationHandler = std::make_shared<
					xolotlCore::DummyTrapMutationHandler>();
		if (!map["attenuation"])
			theTrapMutationHandler->setAttenuation(false);
		if (!map["oneResolution"] && !map["partialResolution"] && !map["fullResolution"])
			theReSolutionHandler = std::make_shared<
					xolotlCore::DummyReSolutionHandler>();
		if (!map["heterogeneous"])
			theNucleationHandler = std::make_shared<
					xolotlCore::DummyNucleationHandler>();

		return;
	}

	/**
	 * Return the flux handler.
	 *
	 *  @return The flux handler.
	 */
	std::shared_ptr<xolotlCore::IFluxHandler> getFluxHandler() const {
		return theFluxHandler;
	}

	/**
	 * Return the advection handler.
	 *
	 *  @return The advection handler.
	 */
	std::vector<std::shared_ptr<xolotlCore::IAdvectionHandler> > getAdvectionHandler() const {
		return theAdvectionHandler;
	}

	/**
	 * Return the diffusion handler.
	 *
	 *  @return The diffusion handler.
	 */
	std::shared_ptr<xolotlCore::IDiffusionHandler> getDiffusionHandler() const {
		return theDiffusionHandler;
	}

	/**
	 * Return the modified trap-mutation handler.
	 *
	 *  @return The trap-mutation handler.
	 */
	std::shared_ptr<xolotlCore::ITrapMutationHandler> getTrapMutationHandler() const {
		return theTrapMutationHandler;
	}

	/**
	 * Return the Re-solution handler.
	 *
	 *  @return The re-solution handler.
	 */
	std::shared_ptr<xolotlCore::IReSolutionHandler> getReSolutionHandler() const {
		return theReSolutionHandler;
	}

	/**
	 * Return the heterogeneous nucleation handler.
	 *
	 *  @return The nucleation handler.
	 */
	std::shared_ptr<xolotlCore::IHeterogeneousNucleationHandler> getNucleationHandler() const {
		return theNucleationHandler;
	}
};

} // end namespace xolotlFactory

#endif // MATERIALHANDLERFACTORY_H
