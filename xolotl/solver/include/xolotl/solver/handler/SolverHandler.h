#pragma once

// Includes
#include <xolotl/core/Constants.h>
#include <xolotl/solver/handler/ISolverHandler.h>
#include <xolotl/util/RandomNumberGenerator.h>

namespace xolotl
{
namespace solver
{
namespace handler
{
/**
 * This class and its subclasses realize the ISolverHandler interface to solve
 * the advection-diffusion-reaction problem with currently supported solvers.
 */
class SolverHandler : public ISolverHandler
{
public:
	using NetworkType = core::network::IReactionNetwork;

	using ConcentrationsView = typename NetworkType::ConcentrationsView;
	using FluxesView = typename NetworkType::FluxesView;
	using SparseFillMap = typename NetworkType::SparseFillMap;

protected:
	/**
	 * The vector to know where the GB are.
	 *
	 * The first pair is the location of a grid point (X,Y),
	 */
	std::vector<std::array<IdType, 3>> gbVector;

	//! The name of the network file
	std::string networkName;

	//! The name of the free GB file
	std::string gbFileName;

	//! The original network created from the network loader.
	NetworkType& network;

	//! Vector storing the grid in the x direction
	std::vector<double> grid;

	//! The number of grid points in the depth direction.
	IdType nX;

	//! The number of grid points in the Y direction.
	IdType nY;

	//! The number of grid points in the Z direction.
	IdType nZ;

	//! The grid step size in the y direction.
	double hY;

	//! The grid step size in the z direction.
	double hZ;

	//! The local start of grid points in the X direction.
	IdType localXS;

	//! The local width of grid points in the X direction.
	IdType localXM;

	//! The local start of grid points in the Y direction.
	IdType localYS;

	//! The local width of grid points in the Y direction.
	IdType localYM;

	//! The local start of grid points in the Z direction.
	IdType localZS;

	//! The local width of grid points in the Z direction.
	IdType localZM;

	//! The number of grid points by which the boundary condition should be
	//! shifted at this side.
	IdType leftOffset, rightOffset, bottomOffset, topOffset, frontOffset,
		backOffset;

	//! The initial vacancy concentration.
	double initialVConc;

	//! The vector of quantities to pass to MOOSE.
	// 0: Xe rate, 1: previous flux, 2: monomer concentration, 3: volume
	// fraction
	std::vector<std::vector<std::vector<std::array<double, 4>>>> localNE;

	//! The electronic stopping power for re-solution
	double electronicStoppingPower;

	//! The original flux handler created.
	core::flux::IFluxHandler* fluxHandler;

	//! The original temperature handler created.
	core::temperature::ITemperatureHandler* temperatureHandler;

	//! the original perf handler created.
	std::shared_ptr<perf::IPerfHandler> perfHandler;

	//! the original viz handler created.
	std::shared_ptr<viz::IVizHandler> vizHandler;

	//! The original diffusion handler created.
	core::diffusion::IDiffusionHandler* diffusionHandler;

	//! The vector of advection handlers.
	std::vector<core::advection::IAdvectionHandler*> advectionHandlers;

	//! The number of dimensions for the problem.
	int dimension;

	//! The portion of void at the beginning of the problem.
	double portion;

	//! If the user wants to move the surface.
	bool movingSurface;

	//! If the user wants to use x mirror boundary conditions or periodic ones.
	bool isMirror;

	//! If the user wants to attenuate the modified trap mutation.
	bool useAttenuation;

	//! The sputtering yield for the problem.
	double sputteringYield;

	//! The ratio of He per V in a bubble.
	double heVRatio;

	//! The value to use to seed the random number generator.
	unsigned int rngSeed;

	//! The minimum sizes for average radius computation.
	std::vector<size_t> minRadiusSizes;

	//! The previous time.
	double previousTime;

	//! The number of xenon atoms that went to the GB
	double nXeGB;

	//! The random number generator to use.
	std::unique_ptr<util::RandomNumberGenerator<int, unsigned int>> rng;

	/**
	 * Method generating the grid in the x direction
	 *
	 * @param opts The options
	 */
	void
	generateGrid(const options::IOptions& opts);

	/**
	 * Constructor.
	 *
	 * @param _network The reaction network to use.
	 * @param options The options.
	 */
	SolverHandler(NetworkType& _network, const options::IOptions& options);

public:
	//! The Constructor
	SolverHandler() = delete;

	virtual ~SolverHandler();

	/**
	 * \see ISolverHandler.h
	 */
	void
	initializeHandlers(core::material::IMaterialHandler* material,
		core::temperature::ITemperatureHandler* tempHandler,
		const options::IOptions& opts) override;

	/**
	 * \see ISolverHandler.h
	 */
	std::vector<double>
	getXGrid() const override
	{
		return grid;
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getStepSizeY() const override
	{
		return hY;
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getStepSizeZ() const override
	{
		return hZ;
	}

	/**
	 * \see ISolverHandler.h
	 */
	int
	getDimension() const override
	{
		return dimension;
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getInitialVConc() const override
	{
		return initialVConc;
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getSputteringYield() const override
	{
		return sputteringYield;
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getHeVRatio() const override
	{
		return heVRatio;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	createLocalNE(IdType a, IdType b = 1, IdType c = 1) override;

	/**
	 * \see ISolverHandler.h
	 */
	void
	setLocalXeRate(double rate, IdType i, IdType j = 0, IdType k = 0) override
	{
		std::get<0>(localNE[i][j][k]) += rate;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setLocalNE(
		const std::vector<std::vector<std::vector<std::array<double, 4>>>>&
			rateVector) override
	{
		localNE = rateVector;
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::vector<std::vector<std::vector<std::array<double, 4>>>>&
	getLocalNE() override
	{
		return localNE;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setPreviousXeFlux(
		double flux, IdType i, IdType j = 0, IdType k = 0) override
	{
		std::get<1>(localNE[i][j][k]) = flux;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setMonomerConc(double conc, IdType i, IdType j = 0, IdType k = 0) override
	{
		std::get<2>(localNE[i][j][k]) = conc;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setVolumeFraction(
		double frac, IdType i, IdType j = 0, IdType k = 0) override
	{
		std::get<3>(localNE[i][j][k]) = frac;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setLocalCoordinates(IdType xs, IdType xm, IdType ys = 0, IdType ym = 0,
		IdType zs = 0, IdType zm = 0) override
	{
		localXS = xs;
		localXM = xm;
		localYS = ys;
		localYM = ym;
		localZS = zs;
		localZM = zm;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	getLocalCoordinates(IdType& xs, IdType& xm, IdType& Mx, IdType& ys,
		IdType& ym, IdType& My, IdType& zs, IdType& zm, IdType& Mz) override
	{
		xs = localXS;
		xm = localXM;
		Mx = nX;
		ys = localYS;
		ym = localYM;
		My = nY;
		zs = localZS;
		zm = localZM;
		Mz = nZ;
	}

	/**
	 * \see ISolverHandler.h
	 */
	IdType
	getLeftOffset() const override
	{
		return leftOffset;
	}

	/**
	 * \see ISolverHandler.h
	 */
	IdType
	getRightOffset() const override
	{
		return rightOffset;
	}

	/**
	 * \see ISolverHandler.h
	 */
	bool
	moveSurface() const override
	{
		return movingSurface;
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getPreviousTime() override
	{
		return previousTime;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setPreviousTime(double time, bool updateFluence = false) override
	{
		previousTime = time;
		if (updateFluence)
			fluxHandler->computeFluence(time);
	}

	/**
	 * \see ISolverHandler.h
	 */
	double
	getNXeGB() override
	{
		return nXeGB;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setNXeGB(double nXe) override
	{
		nXeGB = nXe;
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::vector<size_t>
	getMinSizes() const override
	{
		return minRadiusSizes;
	}

	/**
	 * \see ISolverHandler.h
	 */
	core::flux::IFluxHandler*
	getFluxHandler() const override
	{
		return fluxHandler;
	}

	/**
	 * \see ISolverHandler.h
	 */
	core::temperature::ITemperatureHandler*
	getTemperatureHandler() const override
	{
		return temperatureHandler;
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::shared_ptr<perf::IPerfHandler>
	getPerfHandler() const override
	{
		return perfHandler;
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::shared_ptr<viz::IVizHandler>
	getVizHandler() const override
	{
		return vizHandler;
	}

	/**
	 * \see ISolverHandler.h
	 */
	core::diffusion::IDiffusionHandler*
	getDiffusionHandler() const override
	{
		return diffusionHandler;
	}

	/**
	 * \see ISolverHandler.h
	 */
	core::advection::IAdvectionHandler*
	getAdvectionHandler() const override
	{
		return advectionHandlers[0];
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::vector<core::advection::IAdvectionHandler*>
	getAdvectionHandlers() const override
	{
		return advectionHandlers;
	}

	/**
	 * \see ISolverHandler.h
	 */
	virtual core::network::IReactionNetwork&
	getNetwork() const override
	{
		return network;
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::string
	getNetworkName() const override
	{
		return networkName;
	}

	/**
	 * \see ISolverHandler.h
	 */
	util::RandomNumberGenerator<int, unsigned int>&
	getRNG(void) const override
	{
		return *rng;
	}

	/**
	 * \see ISolverHandler.h
	 */
	std::vector<std::array<IdType, 3>>
	getGBVector() const override
	{
		return gbVector;
	}

	/**
	 * \see ISolverHandler.h
	 */
	void
	setGBLocation(IdType i, IdType j = 0, IdType k = 0) override;

	/**
	 * \see ISolverHandler.h
	 */
	void
	resetGBVector() override
	{
		gbVector.clear();
	}
};
} /* namespace handler */
} /* namespace solver */
} /* namespace xolotl */
