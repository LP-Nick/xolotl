#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Crs.hpp>

#include <plsm/Subpaving.h>
#include <plsm/refine/RegionDetector.h>

#include <xolotl/core/network/Cluster.h>
#include <xolotl/core/network/IReactionNetwork.h>
#include <xolotl/core/network/Reaction.h>
#include <xolotl/core/network/SpeciesEnumSequence.h>
#include <xolotl/core/network/detail/ReactionCollection.h>
#include <xolotl/options/IOptions.h>
#include <xolotl/options/Options.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace detail
{
template <typename TImpl>
struct ReactionNetworkWorker;

template <typename TImpl, typename TDerived>
class ReactionGeneratorBase;
} // namespace detail

template <typename TImpl>
struct ReactionNetworkInterface
{
	using Type = IReactionNetwork;
};

template <typename TImpl>
class ReactionNetwork : public ReactionNetworkInterface<TImpl>::Type
{
	friend class detail::ReactionNetworkWorker<TImpl>;
	template <typename, typename>
	friend class detail::ReactionGeneratorBase;

public:
	using Superclass = typename ReactionNetworkInterface<TImpl>::Type;
	using Traits = ReactionNetworkTraits<TImpl>;
	using Species = typename Traits::Species;

private:
	static_assert(std::is_base_of<IReactionNetwork, Superclass>::value,
		"ReactionNetwork must inherit from IReactionNetwork");

	using Types = detail::ReactionNetworkTypes<TImpl>;

	static constexpr std::size_t numSpecies = Traits::numSpecies;

public:
	using SpeciesSequence = SpeciesEnumSequence<Species, numSpecies>;
	using SpeciesRange = EnumSequenceRange<Species, numSpecies>;
	using ClusterGenerator = typename Traits::ClusterGenerator;
	using ClusterUpdater = typename Types::ClusterUpdater;
	using AmountType = typename IReactionNetwork::AmountType;
	using IndexType = typename IReactionNetwork::IndexType;
	using Subpaving = typename Types::Subpaving;
	using SubdivisionRatio = plsm::SubdivisionRatio<numSpecies>;
	using Composition = typename Types::Composition;
	using Region = typename Types::Region;
	using Ival = typename Region::IntervalType;
	using ConcentrationsView = typename IReactionNetwork::ConcentrationsView;
	using FluxesView = typename IReactionNetwork::FluxesView;
	using SparseFillMap = typename IReactionNetwork::SparseFillMap;
	using ClusterData = typename Types::ClusterData;
	using ClusterDataMirror = typename Types::ClusterDataMirror;
	using ClusterDataRef = typename Types::ClusterDataRef;
	using ReactionCollection = typename Types::ReactionCollection;
	using Bounds = IReactionNetwork::Bounds;
	using PhaseSpace = IReactionNetwork::PhaseSpace;

	template <typename PlsmContext>
	using Cluster = Cluster<TImpl, PlsmContext>;

	ReactionNetwork() = default;

	ReactionNetwork(const Subpaving& subpaving, IndexType gridSize,
		const options::IOptions& opts);

	ReactionNetwork(const Subpaving& subpaving, IndexType gridSize);

	ReactionNetwork(const std::vector<AmountType>& maxSpeciesAmounts,
		const std::vector<SubdivisionRatio>& subdivisionRatios,
		IndexType gridSize, const options::IOptions& opts);

	ReactionNetwork(const std::vector<AmountType>& maxSpeciesAmounts,
		IndexType gridSize, const options::IOptions& opts);

	KOKKOS_INLINE_FUNCTION
	static constexpr std::size_t
	getNumberOfSpecies() noexcept
	{
		return SpeciesSequence::size();
	}

	KOKKOS_INLINE_FUNCTION
	static constexpr std::size_t
	getNumberOfSpeciesNoI() noexcept
	{
		return SpeciesSequence::sizeNoI();
	}

	KOKKOS_INLINE_FUNCTION
	static constexpr SpeciesRange
	getSpeciesRange() noexcept
	{
		return SpeciesRange{};
	}

	KOKKOS_INLINE_FUNCTION
	static constexpr SpeciesRange
	getSpeciesRangeNoI() noexcept
	{
		return SpeciesRange(
			SpeciesSequence::first(), SpeciesSequence::lastNoI());
	}

	KOKKOS_INLINE_FUNCTION
	static constexpr SpeciesRange
	getSpeciesRangeForGrouping() noexcept
	{
		using GroupingRange =
			SpeciesForGrouping<Species, SpeciesSequence::size()>;
		return SpeciesRange(GroupingRange::first, GroupingRange::last);
	}

	KOKKOS_INLINE_FUNCTION
	static constexpr std::underlying_type_t<Species>
	mapToMomentId(EnumSequence<Species, SpeciesSequence::size()> value)
	{
		using GroupingRange =
			SpeciesForGrouping<Species, SpeciesSequence::size()>;
		return GroupingRange::mapToMomentId(value);
	}

	void
	initializeExtraClusterData(const options::IOptions&)
	{
	}

	void
	updateExtraClusterData(const std::vector<double>&)
	{
	}

	std::size_t
	getSpeciesListSize() const noexcept override
	{
		return getNumberOfSpecies();
	}

	const std::string&
	getSpeciesLabel(SpeciesId id) const override;

	const std::string&
	getSpeciesName(SpeciesId id) const override;

	SpeciesId
	parseSpeciesId(const std::string& speciesLabel) const override;

	void
	setLatticeParameter(double latticeParameter) override;

	void
	setImpurityRadius(double impurityRadius) noexcept override
	{
		this->_impurityRadius =
			asDerived()->checkImpurityRadius(impurityRadius);
	}

	void
	setFissionRate(double rate) override;

	void
	setZeta(double zeta) override;

	void
	setTauBursting(double tau) override;

	void
	setFBursting(double f) override;

	void
	setEnableStdReaction(bool reaction) override;

	void
	setEnableReSolution(bool reaction) override;

	void
	setEnableNucleation(bool reaction) override;

	void
	setEnableSink(bool reaction) override;

	void
	setEnableTrapMutation(bool reaction) override;

	void
	setEnableBursting(bool reaction) override;

	void
	setEnableReducedJacobian(bool reduced) override;

	void
	setGridSize(IndexType gridSize) override;

	void
	setTemperatures(const std::vector<double>& gridTemperatures) override;

	std::uint64_t
	getDeviceMemorySize() const noexcept override;

	void
	syncClusterDataOnHost() override;

	KOKKOS_INLINE_FUNCTION
	Cluster<plsm::OnDevice>
	findCluster(const Composition& comp, plsm::OnDevice context);

	Cluster<plsm::OnHost>
	findCluster(const Composition& comp, plsm::OnHost context);

	KOKKOS_INLINE_FUNCTION
	auto
	findCluster(const Composition& comp)
	{
		return findCluster(comp, plsm::onDevice);
	}

	IndexType
	findClusterId(const std::vector<AmountType>& composition) override
	{
		assert(composition.size() == getNumberOfSpecies());
		Composition comp;
		for (std::size_t i = 0; i < composition.size(); ++i) {
			comp[i] = composition[i];
		}
		return findCluster(comp, plsm::onHost).getId();
	}

	ClusterCommon<plsm::OnHost>
	getClusterCommon(IndexType clusterId) const override
	{
		return ClusterCommon<plsm::OnHost>(_clusterDataMirror, clusterId);
	}

	ClusterCommon<plsm::OnHost>
	getSingleVacancy() override;

	IndexType
	getLargestClusterId() override
	{
		return asDerived()->checkLargestClusterId();
	}

	Bounds
	getAllClusterBounds() override;

	PhaseSpace
	getPhaseSpace() override;

	KOKKOS_INLINE_FUNCTION
	Cluster<plsm::OnDevice>
	getCluster(IndexType clusterId, plsm::OnDevice)
	{
		return Cluster<plsm::OnDevice>(_clusterData, clusterId);
	}

	Cluster<plsm::OnHost>
	getCluster(IndexType clusterId, plsm::OnHost)
	{
		return Cluster<plsm::OnHost>(_clusterDataMirror, clusterId);
	}

	KOKKOS_INLINE_FUNCTION
	auto
	getCluster(IndexType clusterId)
	{
		return getCluster(clusterId, plsm::onDevice);
	}

	KOKKOS_INLINE_FUNCTION
	Subpaving&
	getSubpaving()
	{
		return _subpaving;
	}

	void
	computeFluxesPreProcess(
		ConcentrationsView, FluxesView, IndexType, double, double)
	{
	}

	void
	computeAllFluxes(ConcentrationsView concentrations, FluxesView fluxes,
		IndexType gridIndex = 0, double surfaceDepth = 0.0,
		double spacing = 0.0) final;

	template <typename TReaction>
	void
	computeFluxes(ConcentrationsView concentrations, FluxesView fluxes,
		IndexType gridIndex = 0, double surfaceDepth = 0.0,
		double spacing = 0.0)
	{
		asDerived()->computeFluxesPreProcess(
			concentrations, fluxes, gridIndex, surfaceDepth, spacing);

		_reactions.template forEachOn<TReaction>(
			DEVICE_LAMBDA(auto&& reaction) {
				reaction.contributeFlux(concentrations, fluxes, gridIndex);
			});
		Kokkos::fence();
	}

	void
	computePartialsPreProcess(
		ConcentrationsView, FluxesView, IndexType, double, double)
	{
	}

	void
	computeAllPartials(ConcentrationsView concentrations,
		Kokkos::View<double*> values, IndexType gridIndex = 0,
		double surfaceDepth = 0.0, double spacing = 0.0) override;

	template <typename TReaction>
	void
	computePartials(ConcentrationsView concentrations,
		Kokkos::View<double*> values, IndexType gridIndex = 0,
		double surfaceDepth = 0.0, double spacing = 0.0)
	{
		// Reset the values
		const auto& nValues = values.extent(0);
		Kokkos::parallel_for(
			nValues, KOKKOS_LAMBDA(const IndexType i) { values(i) = 0.0; });

		asDerived()->computePartialsPreProcess(
			concentrations, values, gridIndex, surfaceDepth, spacing);

		auto connectivity = _reactions.getConnectivity();
		if (this->_enableReducedJacobian) {
			_reactions.template forEachOn<TReaction>(
				DEVICE_LAMBDA(auto&& reaction) {
					reaction.contributeReducedPartialDerivatives(
						concentrations, values, connectivity, gridIndex);
				});
		}
		else {
			_reactions.template forEachOn<TReaction>(
				DEVICE_LAMBDA(auto&& reaction) {
					reaction.contributePartialDerivatives(
						concentrations, values, connectivity, gridIndex);
				});
		}

		Kokkos::fence();
	}

	double
	getLargestRate() override;

	double
	getLeftSideRate(ConcentrationsView concentrations, IndexType clusterId,
		IndexType gridIndex) override;

	IndexType
	getDiagonalFill(SparseFillMap& fillMap) override;

	/**
	 * Get the total concentration of a given type of clusters.
	 *
	 * @param concentration The vector of concentrations
	 * @param type The type of atom we want the concentration of
	 * @param minSize The minimum number of atom to start counting
	 * @return The total accumulated concentration
	 */
	double
	getTotalConcentration(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	double
	getTotalConcentration(ConcentrationsView concentrations, SpeciesId species,
		AmountType minSize = 0) override
	{
		auto type = species.cast<Species>();
		return getTotalConcentration(concentrations, type, minSize);
	}

	/**
	 * Get the total concentration of a given type of clusters times their
	 * radius.
	 *
	 * @param concentration The vector of concentrations
	 * @param type The type of atom we want the concentration of
	 * @param minSize The minimum number of atom to start counting
	 * @return The total accumulated concentration times the radius of the
	 * cluster
	 */
	double
	getTotalRadiusConcentration(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	double
	getTotalRadiusConcentration(ConcentrationsView concentrations,
		SpeciesId species, AmountType minSize = 0) override
	{
		auto type = species.cast<Species>();
		return getTotalRadiusConcentration(concentrations, type, minSize);
	}

	/**
	 * Get the total concentration of a given type of clusters times the number
	 * of atoms.
	 *
	 * @param concentration The vector of concentrations
	 * @param type The type of atom we want the concentration of
	 * @param minSize The minimum number of atom to start counting
	 * @return The total accumulated concentration times the size of the cluster
	 */
	double
	getTotalAtomConcentration(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	double
	getTotalAtomConcentration(ConcentrationsView concentrations,
		SpeciesId species, AmountType minSize = 0) override
	{
		auto type = species.cast<Species>();
		return getTotalAtomConcentration(concentrations, type, minSize);
	}

	/**
	 * Get the total concentration of a given type of clusters only if it is
	 * trapped in a vacancy.
	 *
	 * @param concentration The vector of concentrations
	 * @param type The type of atom we want the concentration of
	 * @param minSize The minimum number of atom to start counting
	 * @return The total accumulated concentration times the size of the cluster
	 */
	double
	getTotalTrappedAtomConcentration(ConcentrationsView concentrations,
		Species type, AmountType minSize = 0);

	/**
	 * Get the total concentration of a given type of clusters times their
	 * volume.
	 *
	 * @param concentration The vector of concentrations
	 * @param type The type of atom we want the concentration of
	 * @param minSize The minimum number of atom to start counting
	 * @return The total accumulated volume fraction
	 */
	double
	getTotalVolumeFraction(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	void
	updateReactionRates();

	void
	updateOutgoingDiffFluxes(double* gridPointSolution, double factor,
		std::vector<IndexType> diffusingIds, std::vector<double>& fluxes,
		IndexType gridIndex) override;

	void
	updateOutgoingAdvecFluxes(double* gridPointSolution, double factor,
		std::vector<IndexType> advectingIds, std::vector<double> sinkStrengths,
		std::vector<double>& fluxes, IndexType gridIndex) override;

private:
	KOKKOS_INLINE_FUNCTION
	TImpl*
	asDerived()
	{
		return static_cast<TImpl*>(this);
	}

	static std::map<std::string, SpeciesId>
	createSpeciesLabelMap() noexcept;

	void
	defineMomentIds();

	void
	generateClusterData(const ClusterGenerator& generator);

	void
	defineReactions();

	void
	updateDiffusionCoefficients();

	KOKKOS_INLINE_FUNCTION
	double
	getTemperature(IndexType gridIndex) const noexcept
	{
		return _clusterData.temperature(gridIndex);
	}

private:
	Subpaving _subpaving;
	ClusterDataMirror _clusterDataMirror;

	detail::ReactionNetworkWorker<TImpl> _worker;

protected:
	ClusterData _clusterData;

	ReactionCollection _reactions;

	std::map<std::string, SpeciesId> _speciesLabelMap;
};

namespace detail
{
template <typename TImpl>
struct ReactionNetworkWorker
{
	using Network = ReactionNetwork<TImpl>;
	using Types = ReactionNetworkTypes<TImpl>;
	using Species = typename Types::Species;
	using ClusterData = typename Types::ClusterData;
	using ClusterDataRef = typename Types::ClusterDataRef;
	using IndexType = typename Types::IndexType;
	using AmountType = typename Types::AmountType;
	using ReactionCollection = typename Types::ReactionCollection;
	using ConcentrationsView = typename IReactionNetwork::ConcentrationsView;

	Network& _nw;

	ReactionNetworkWorker(Network& network) : _nw(network)
	{
	}

	void
	updateDiffusionCoefficients();

	void
	defineMomentIds();

	void
	defineReactions();

	IndexType
	getDiagonalFill(typename Network::SparseFillMap& fillMap);

	double
	getTotalConcentration(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	double
	getTotalAtomConcentration(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	double
	getTotalRadiusConcentration(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);

	double
	getTotalVolumeFraction(ConcentrationsView concentrations, Species type,
		AmountType minSize = 0);
};

template <typename TImpl>
class DefaultClusterUpdater
{
public:
	using Network = ReactionNetwork<TImpl>;
	using Types = ReactionNetworkTypes<TImpl>;
	using ClusterData = typename Types::ClusterData;
	using IndexType = typename Types::IndexType;

	KOKKOS_INLINE_FUNCTION
	void
	updateDiffusionCoefficient(const ClusterData& data, IndexType clusterId,
		IndexType gridIndex) const;
};
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
