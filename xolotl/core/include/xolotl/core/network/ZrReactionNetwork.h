#pragma once

#include <xolotl/core/Constants.h>
#include <xolotl/core/network/ReactionNetwork.h>
#include <xolotl/core/network/ZrReaction.h>
#include <xolotl/core/network/ZrTraits.h>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace detail
{
class ZrReactionGenerator;

class ZrClusterUpdater;
} // namespace detail

class ZrReactionNetwork : public ReactionNetwork<ZrReactionNetwork>
{
	friend class ReactionNetwork<ZrReactionNetwork>;

public:
	using Superclass = ReactionNetwork<ZrReactionNetwork>;
	using Subpaving = typename Superclass::Subpaving;
	using Composition = typename Superclass::Composition;
	using Species = typename Superclass::Species;
	using IndexType = typename Superclass::IndexType;
	using ConcentrationsView = typename Superclass::ConcentrationsView;
	using FluxesView = typename Superclass::FluxesView;
	using RateVector = typename Superclass::RateVector;
	using ConnectivitiesPair = typename Superclass::ConnectivitiesPair;
	using RatesView = typename Superclass::RatesView;

	using Superclass::Superclass;

	IndexType
	checkLargestClusterId();
	
	IndexType
	getLargestClusterId()
	{
		return largestClusterId;
	}
	
	void
	setConstantRates(RatesView rates, IndexType gridIndex) override;

	void
	setConstantConnectivities(ConnectivitiesPair conns) override;

	void
	setConstantRateEntries() override;

	void
	initializeExtraClusterData(const options::IOptions& options);

	void
	updateExtraClusterData(const std::vector<double>& gridTemps,
		const std::vector<double>& gridDepths);

	std::vector<double>
	calcThermalRadii(const std::vector<std::vector<double>>& alphaDVec, 
		const std::vector<std::vector<double>>& mDVec, const double& temp, const double& size, const int& species);

	void
	setGridSize(IndexType gridSize) override;

	std::string
	getMonitorOutputFileName() const override
	{
		return "AlphaZr.dat";
	}

	std::string
	getMonitorDataHeaderString() const override;

	void
	addMonitorDataValues(Kokkos::View<const double*> conc, double fac,
		std::vector<double>& totalVals) override;

	std::size_t
	getMonitorDataLineSize() const override
	{
		return getSpeciesListSize() * 6;
	}

	void
	writeMonitorDataLine(
		const std::vector<double>& localData, double time) override;

	std::string
	getRxnOutputFileName() const override
	{
		return "Rxn.dat";
	}

	std::string
	getRxnDataHeaderString() const override;

	void
	addRxnDataValues(Kokkos::View<const double*> conc,
		std::vector<std::vector<double>>& totalVals) override;

	std::size_t
	getRxnDataLineSize() const override
	{
		return 35;
	}

	void
	writeRxnDataLine(const std::vector<std::vector<double>>& localData,
		double time) override;
		
	void
	initializeExtraDOFs(const options::IOptions& options);

	void
	computeFluxesPreProcess(ConcentrationsView concentrations,
		FluxesView fluxes, IndexType gridIndex, double surfaceDepth,
		double spacing);

	void
	computePartialsPreProcess(ConcentrationsView concentrations,
		Kokkos::View<double*> values, IndexType gridIndex, double surfaceDepth,
		double spacing);

	double
	computeClusterRadius(double amount, int species)
	{
		//Find the edge of the phase space
		const auto& largestReg = this->getCluster(largestClusterId).getRegion();
		Composition hiLargest = largestReg.getUpperLimitPoint();
		double largestSize = hiLargest[Species::V] + hiLargest[Species::I] +
													hiLargest[Species::Basal] - 3; //dont know which one was saved
		amount = util::max(amount, largestSize);
		if (species == 0){
			//Vac case
			return pow(amount+1.0, 1.0/2.0) * pow(3.23*5.17/(2.0* ::xolotl::core::pi), 1.0/2.0) * 1.118 / 10.0;
		}
		if (species == 1){
			//Int case
			return pow(amount+1.0, 1.0/2.0) * pow(3.23*5.17/(2.0* ::xolotl::core::pi), 1.0/2.0) * 1.026 / 10.0;
		}
		if (species == 2){
			//Basal case
			return pow(amount+1.0, 1.0/2.0) * 3.23 * pow(pow(3.0, 1.0/2.0)/(2.0* ::xolotl::core::pi), 1.0/2.0) / 10.0;
		}
	}
public:
	IndexType largestClusterId;
	
private:
	double
	checkLatticeParameter(double latticeParameter);

	double
	computeAtomicVolume(double latticeParameter)
	{
		// sqrt(3) / 4 * a^2 * c
		// with c the other lattice parameter
		return 0.0234; // nm^3
	}

	double
	checkImpurityRadius(double impurityRadius);

	detail::ZrReactionGenerator
	getReactionGenerator() const noexcept;

	void
	readClusters(const std::string filename)
	{
		return;
	}

	void
	readReactions(double temperature, const std::string filename)
	{
		return;
	}

	void
	defineReactions(Connectivity& connectivity);
};

namespace detail
{
class ZrReactionGenerator :
	public ReactionGenerator<ZrReactionNetwork, ZrReactionGenerator>
{
	friend class ReactionGeneratorBase<ZrReactionNetwork, ZrReactionGenerator>;

public:
	using Network = ZrReactionNetwork;
	using Subpaving = typename Network::Subpaving;
	using Superclass =
		ReactionGenerator<ZrReactionNetwork, ZrReactionGenerator>;

	using Superclass::Superclass;

	ZrReactionGenerator(const ZrReactionNetwork& network) :
		Superclass(network),
		largestClusterId(network.largestClusterId)
		{
		}

	template <typename TTag>
	KOKKOS_INLINE_FUNCTION
	void
	operator()(IndexType i, IndexType j, TTag tag) const;

	template <typename TTag>
	KOKKOS_INLINE_FUNCTION
	void
	addSinks(IndexType i, TTag tag) const;
	
	template <typename TTag>
	
	KOKKOS_INLINE_FUNCTION
	void
	addSingleSizeReactions(IndexType i, IndexType j, TTag tag) const;
	

private:
	ReactionCollection<Network>
	getReactionCollection() const;
	
	IndexType largestClusterId;
};

class ZrClusterUpdater
{
public:
	using Network = ZrReactionNetwork;
	using ClusterData = typename Network::ClusterData;
	using IndexType = typename Network::IndexType;

	KOKKOS_INLINE_FUNCTION
	void
	updateDiffusionCoefficient(const ClusterData& data, IndexType clusterId,
		IndexType gridIndex) const;
};
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl

#include <xolotl/core/network/ZrClusterGenerator.h>

#if defined(XOLOTL_INCLUDE_RN_TPP_FILES)
#include <xolotl/core/network/impl/ZrReactionNetwork.tpp>
#endif
