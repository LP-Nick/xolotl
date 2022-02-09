#pragma once

#include <plsm/refine/Detector.h>

#include <xolotl/core/network/FeTraits.h>

namespace xolotl
{
namespace core
{
namespace network
{
class FeClusterGenerator : public plsm::refine::Detector<FeClusterGenerator>
{
public:
	using Species = FeSpeciesList;
	using Superclass = plsm::refine::Detector<FeClusterGenerator>;
	using NetworkType = FeReactionNetwork;

	template <typename PlsmContext>
	using Cluster = typename NetworkType::Cluster<PlsmContext>;

	using Region = typename NetworkType::Region;
	using Composition = typename NetworkType::Composition;
	using AmountType = typename NetworkType::AmountType;
	using BoolArray = plsm::refine::BoolVec<Region>;

	FeClusterGenerator(const options::IOptions& options);

	FeClusterGenerator(
		const options::IOptions& options, std::size_t refineDepth);

	KOKKOS_INLINE_FUNCTION
	bool
	refine(const Region& region, BoolArray& result) const;

	KOKKOS_INLINE_FUNCTION
	bool
	select(const Region& region) const;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getFormationEnergy(const Cluster<PlsmContext>& cluster) const noexcept
	{
		// Always return 0.0 here because we use capillarity laws for the
		// binding energies
		return 0.0;
	}

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getMigrationEnergy(const Cluster<PlsmContext>& cluster) const noexcept;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getDiffusionFactor(const Cluster<PlsmContext>& cluster,
		double latticeParameter) const noexcept;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getReactionRadius(const Cluster<PlsmContext>& cluster,
		double latticeParameter, double interstitialBias,
		double impurityRadius) const noexcept;

private:
	KOKKOS_INLINE_FUNCTION
	double
	getHeVFormationEnergy(Composition comp) const noexcept;

private:
	// Maximum size of single species
	AmountType _maxHe{8};
	AmountType _maxV{0};
	AmountType _groupingMin;
	AmountType _groupingWidthHe;
	AmountType _groupingWidthV;
};
} // namespace network
} // namespace core
} // namespace xolotl
