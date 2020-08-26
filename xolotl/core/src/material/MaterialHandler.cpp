#include <xolotl/core/InvalidGridDimension.h>
#include <xolotl/core/advection/XGBAdvectionHandler.h>
#include <xolotl/core/advection/YGBAdvectionHandler.h>
#include <xolotl/core/advection/ZGBAdvectionHandler.h>
#include <xolotl/core/diffusion/Diffusion1DHandler.h>
#include <xolotl/core/diffusion/Diffusion2DHandler.h>
#include <xolotl/core/diffusion/Diffusion3DHandler.h>
#include <xolotl/core/diffusion/DummyDiffusionHandler.h>
#include <xolotl/core/material/MaterialHandler.h>
#include <xolotl/util/TokenizedLineReader.h>

namespace xolotl
{
namespace core
{
namespace material
{
MaterialHandler::MaterialHandler(const options::Options& options,
	const IMaterialSubHandlerGenerator& subHandlerGenerator) :
	_diffusionHandler(createDiffusionHandler(options)),
	_advectionHandlers({subHandlerGenerator.generateAdvectionHandler()}),
	_fluxHandler(subHandlerGenerator.generateFluxHandler(options)),
	_trapMutationHandler(subHandlerGenerator.generateTrapMutationHandler())
{
	initializeTrapMutationHandler(options);
	initializeAdvectionHandlers(options);
}

std::shared_ptr<core::diffusion::IDiffusionHandler>
MaterialHandler::createDiffusionHandler(const options::Options& options)
{
	double migrationThreshold = options.getMigrationThreshold();

	if (!options.getProcesses().at("diff")) {
		return std::make_shared<core::diffusion::DummyDiffusionHandler>(
			migrationThreshold);
	}

	switch (options.getDimensionNumber()) {
	case 0:
		return std::make_shared<core::diffusion::DummyDiffusionHandler>(
			migrationThreshold);
		break;
	case 1:
		return std::make_shared<core::diffusion::Diffusion1DHandler>(
			migrationThreshold);
		break;
	case 2:
		return std::make_shared<core::diffusion::Diffusion2DHandler>(
			migrationThreshold);
		break;
	case 3:
		return std::make_shared<core::diffusion::Diffusion3DHandler>(
			migrationThreshold);
		break;
	default:
		// The asked dimension is not good (e.g. -1, 4)
		throw InvalidGridDimension(
			"\nxolotlFactory: Bad dimension for the material factory.");
	}
}

void
MaterialHandler::initializeTrapMutationHandler(const options::Options& options)
{
	if (!options.getProcesses().at("modifiedTM")) {
		_trapMutationHandler =
			std::make_shared<core::modified::DummyTrapMutationHandler>();
	}
	if (!options.getProcesses().at("attenuation")) {
		_trapMutationHandler->setAttenuation(false);
	}
}

void
MaterialHandler::initializeAdvectionHandlers(const options::Options& options)
{
	if (!options.getProcesses().at("advec")) {
		_advectionHandlers.clear();
		_advectionHandlers.push_back(
			std::make_shared<core::advection::DummyAdvectionHandler>());
	}

	// Get the number of dimensions
	auto dim = options.getDimensionNumber();

	// Setup the grain boundaries
	std::string gbString = options.getGbString();
	util::TokenizedLineReader<std::string> reader;
	reader.setInputStream(std::make_shared<std::istringstream>(gbString));
	auto tokens = reader.loadLine();
	for (int i = 0; i < tokens.size(); ++i) {
		if (tokens[i] == "X") {
			_advectionHandlers.push_back(
				std::make_shared<core::advection::XGBAdvectionHandler>());
			_advectionHandlers.back()->setLocation(
				std::strtod(tokens[i + 1].c_str(), nullptr));
			_advectionHandlers.back()->setDimension(dim);
		}
		else if (tokens[i] == "Y") {
			if (dim < 2) {
				throw std::runtime_error(
					"\nA Y grain boundary CANNOT be used "
					"in 1D. Switch to 2D or 3D or remove it.");
			}
			_advectionHandlers.push_back(
				std::make_shared<core::advection::YGBAdvectionHandler>());
			_advectionHandlers.back()->setLocation(
				std::strtod(tokens[i + 1].c_str(), nullptr));
			_advectionHandlers.back()->setDimension(dim);
		}
		else if (tokens[i] == "Z") {
			if (dim < 2) {
				throw std::runtime_error(
					"\nA Z grain boundary CANNOT be used "
					"in 1D/2D. Switch to 3D or remove it.");
			}
			_advectionHandlers.push_back(
				std::make_shared<core::advection::ZGBAdvectionHandler>());
			_advectionHandlers.back()->setLocation(
				std::strtod(tokens[i + 1].c_str(), nullptr));
			_advectionHandlers.back()->setDimension(dim);
		}
		else {
			throw std::runtime_error(
				"\nThe type of grain boundary is not known: \"" + tokens[i] +
				"\"");
		}

		++i;
	}
}
} // namespace material
} // namespace core
} // namespace xolotl
