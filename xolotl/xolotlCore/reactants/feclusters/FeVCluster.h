#ifndef FEVCLUSTER_H
#define FEVCLUSTER_H

// Includes
#include "FeCluster.h"
#include <xolotlPerf.h>
#include <Constants.h>

namespace xolotlCore {

/**
 * This class represents a cluster composed entirely of atomic vacancies.
 */
class FeVCluster: public FeCluster {

private:
	static std::string buildName(IReactant::SizeType nV) {
		std::stringstream nameStream;
		nameStream << "V_" << nV;
		return nameStream.str();
	}

	/**
	 * The sink strength
	 */
	double sinkStrength;

public:

	/**
	 * Default constructor, deleted because we require info to construct.
	 */
	FeVCluster() = delete;

	/**
	 * The constructor. All FeVClusters must be initialized with a size.
	 *
	 * @param nV the number of atomic vacancies in the cluster
	 * @param registry The performance handler registry
	 */
	FeVCluster(int nV, IReactionNetwork& _network,
			std::shared_ptr<xolotlPerf::IHandlerRegistry> registry) :
			FeCluster(_network, registry, buildName(nV)) {

		// Set the size
		size = nV;
		// Set the typename appropriately
		type = ReactantType::V;

		// Update the composition map
		composition[toCompIdx(Species::V)] = size;

		// Compute the reaction radius
		double latticeParam = network.getLatticeParameter();
		// It is the same formula for HeV clusters
		reactionRadius = latticeParam
				* pow((3.0 * size) / xolotlCore::pi, (1.0 / 3.0)) * 0.5;

		// Bounds on He and V
		heBounds = IntegerRange<IReactant::SizeType>(
				static_cast<IReactant::SizeType>(0),
				static_cast<IReactant::SizeType>(1));
		vBounds = IntegerRange<IReactant::SizeType>(
				static_cast<IReactant::SizeType>(size),
				static_cast<IReactant::SizeType>(size + 1));

		//! Parameters for biased sink in the iron case
		double r0 = latticeParam * 0.75 * sqrt(3.0);
		double reactionRadius = latticeParam * cbrt(3.0 / xolotlCore::pi) * 0.5;
		constexpr double rho = 0.0003;
		sinkStrength = -4.0 * xolotlCore::pi * rho
				/ log(
						xolotlCore::pi * rho * (reactionRadius + r0)
								* (reactionRadius + r0));

		return;
	}

	/**
	 * Copy constructor, deleted to prevent use.
	 */
	FeVCluster(const FeVCluster& other) = delete;

	//! Destructor
	~FeVCluster() {
	}

	/**
	 * This operation returns the total change in this cluster due its
	 * own dissociation.
	 *
	 * @param i The location on the grid in the depth direction
	 * @return The flux due to its dissociation
	 */
	double getEmissionFlux(int i) override {
		// Initial declarations
		double flux = FeCluster::getEmissionFlux(i);

		// Compute the loss to dislocation sinks
		if (size < 5) {
			// k^2 * D * C
			flux += sinkStrength * diffusionCoefficient[i]
					* concentration;
		}

		return flux;
	}

	/**
	 * This operation computes the partial derivatives due to emission
	 * reactions.
	 *
	 * @param partials The vector into which the partial derivatives should be
	 * inserted. This vector should have a length equal to the size of the
	 * network.
	 * @param i The location on the grid in the depth direction
	 */
	void getEmissionPartialDerivatives(std::vector<double> & partials,
			int i) const override {
		// Initial declarations
		FeCluster::getEmissionPartialDerivatives(partials, i);

		// Compute the loss to dislocation sinks
		if (size < 5) {
			// k^2 * D * C
			partials[id - 1] -= sinkStrength * diffusionCoefficient[i];
		}

		return;
	}

	/**
	 * Add grid points to the vector of diffusion coefficients or remove
	 * them if the value is negative.
	 *
	 * @param i The number of grid point to add or remove
	 */
	void addGridPoints(int i) override {
		if (diffusionFactor > 0.0) {
			Reactant::addGridPoints(i);
		}

		// Don't do anything
		return;
	}

	/**
	 * This operation sets the temperature at which the reactant currently
	 * exists. Temperature-dependent quantities are recomputed when this
	 * operation is called, so the temperature should always be set first.
	 *
	 * @param temp The new cluster temperature
	 * @param i The location on the grid
	 */
	void setTemperature(double temp, int i) override {
		if (diffusionFactor > 0.0) {
			Reactant::setTemperature(temp, i);
		}

		// Don't do anything
		return;
	}

	/**
	 * This operation returns the diffusion coefficient for this reactant and is
	 * calculated from the diffusion factor.
	 *
	 * @param i The position on the grid
	 * @return The diffusion coefficient
	 */
	double getDiffusionCoefficient(int i) const override {
		if (diffusionFactor > 0.0) {
			return Reactant::getDiffusionCoefficient(i);
		}

		return 0.0;
	}

};
//end class FeVCluster

} /* end namespace xolotlCore */

#endif
