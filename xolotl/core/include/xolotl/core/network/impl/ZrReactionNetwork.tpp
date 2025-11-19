#pragma once

#include <xolotl/core/network/detail/impl/ConstantReactionGenerator.tpp>
#include <xolotl/core/network/detail/impl/SinkReactionGenerator.tpp>
#include <xolotl/core/network/impl/ReactionNetwork.tpp>
#include <xolotl/core/network/impl/ZrClusterGenerator.tpp>
#include <xolotl/core/network/impl/ZrReaction.tpp>

namespace xolotl
{
namespace core
{
namespace network
{

void
ZrReactionNetwork::initializeExtraDOFs(const options::IOptions& options)
{
	auto map = options.getProcesses();
	if (not map["largeCluster"])
		return;
	largestClusterId = checkLargestClusterId();
	
	this->_clusterData.h_view().setVacId(this->_numDOFs);
	this->_clusterData.h_view().setVacAvId(this->_numDOFs + 1);
	this->_clusterData.h_view().setIntId(this->_numDOFs + 2);
	this->_clusterData.h_view().setIntAvId(this->_numDOFs + 3);
	this->_clusterData.h_view().setBasalId(this->_numDOFs + 4);
	this->_clusterData.h_view().setBasalAvId(this->_numDOFs + 5);
	this->_numDOFs +=6;
}

void
ZrReactionNetwork::computeFluxesPreProcess(ConcentrationsView concentrations,
	FluxesView fluxes, IndexType gridIndex, double surfaceDepth, double spacing)
{
	if (this->_enableLargeCluster) {
		auto clusterDataMirror = this->getClusterDataMirror();

		// Get the concentrations on the host
		auto dConcs = Kokkos::subview(concentrations,
			std::make_pair(
				clusterDataMirror.vacId(), clusterDataMirror.basalId() + 1));
		auto hConcs = create_mirror_view(dConcs);
		deep_copy(hConcs, dConcs);

		// Compute the average composition of each defect
		for (int i; i < 3; i++) {
			auto conc = hConcs(2 * i);
			auto avComp = hConcs(2 * i + 1) / conc;
			if (conc == 0.0)
				avComp = 0.0;
			// Compute and save the radius from that
			switch (i) {
			// Vac
			case 0:
				this->_clusterData.h_view().setVacAvRad(util::max(0.0,
					computeClusterRadius(
						avComp, i)));
			// Int
			case 1:
				this->_clusterData.h_view().setIntAvRad(util::max(0.0,
					computeClusterRadius(
						avComp, i)));
			// Basal
			case 2:
				this->_clusterData.h_view().setBasalAvRad(util::max(0.0,
					computeClusterRadius(
						avComp, i)));
			}
		}
	}
}

void
ZrReactionNetwork::computePartialsPreProcess(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	IndexType gridIndex, double surfaceDepth, double spacing)
{
	if (this->_enableLargeCluster) {
		auto clusterDataMirror = this->getClusterDataMirror();

		// Get the concentrations on the host
		auto dConcs = Kokkos::subview(concentrations,
			std::make_pair(
				clusterDataMirror.vacId(), clusterDataMirror.basalId() + 1));
		auto hConcs = create_mirror_view(dConcs);
		deep_copy(hConcs, dConcs);

		// Compute the average composition of each defect
		for (int i; i < 3; i++) {
			auto conc = hConcs(2 * i);
			auto avComp = hConcs(2 * i + 1) / conc;
			if (conc == 0.0)
				avComp = 0.0;
			// Compute and save the radius from that
			switch (i) {
			// Vac
			case 0:
				this->_clusterData.h_view().setVacAvRad(util::max(0.0,
					computeClusterRadius(
						avComp, i)));
			// Int
			case 1:
				this->_clusterData.h_view().setIntAvRad(util::max(0.0,
					computeClusterRadius(
						avComp, i)));
			// Basal
			case 2:
				this->_clusterData.h_view().setBasalAvRad(util::max(0.0,
					computeClusterRadius(
						avComp, i)));
			}
		}
	}
}


namespace detail
{
template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
ZrReactionGenerator::operator()(IndexType i, IndexType j, TTag tag) const
{
	using Species = typename Network::Species;
	using Composition = typename Network::Composition;
	using AmountType = typename Network::AmountType;

	// Get the diffusion factors
	auto diffusionFactor = this->_clusterData.diffusionFactor;

	if (i == j) {
		if (diffusionFactor(i) != 0.0)
			addSinks(i, tag);

		if (this->_constantConnsRows.extent(0) > 0) {
			// Look for the entry
			for (auto k = this->_constantConnsRows(i);
				 k < this->_constantConnsRows(i + 1); k++) {
				if (this->_constantConnsEntries(k) == this->_numDOFs) {
					this->addConstantReaction(
						tag, {i, Network::invalidIndex()});
					break;
				}
			}
		}
	}

	// Add every possibility
	if (this->_constantConnsRows.extent(0) > 0) {
		// Look for the entry
		for (auto k = this->_constantConnsRows(i);
			 k < this->_constantConnsRows(i + 1); k++) {
			if (this->_constantConnsEntries(k) == j) {
				this->addConstantReaction(tag, {i, j});
				break;
			}
		}
	}
	if (j != i) {
		if (this->_constantConnsRows.extent(0) > 0) {
			// Look for the entry
			for (auto k = this->_constantConnsRows(j);
				 k < this->_constantConnsRows(j + 1); k++) {
				if (this->_constantConnsEntries(k) == i) {
					this->addConstantReaction(tag, {j, i});
					break;
				}
			}
		}
	}

	auto& subpaving = this->getSubpaving();
	auto previousIndex = subpaving.invalidIndex();

// Get the composition of each cluster
	const auto& cl1Reg = this->getCluster(i).getRegion();
	const auto& cl2Reg = this->getCluster(j).getRegion();
	Composition lo1 = cl1Reg.getOrigin();
	Composition hi1 = cl1Reg.getUpperLimitPoint();
	Composition lo2 = cl2Reg.getOrigin();
	Composition hi2 = cl2Reg.getUpperLimitPoint();
	
if (lo1[Species::V] > 0){
			std::cout<<"diffusion factor for v"<<lo1[Species::V]<<": " << diffusionFactor(i)<<std::endl;
			std::cout<<std::endl;
	}
if (lo2[Species::V] > 0){
			std::cout<<"diffusion factor for v"<<lo2[Species::V]<<": " << diffusionFactor(j)<<std::endl;
			std::cout<<std::endl;
	}


	// Check the diffusion factors
	if (diffusionFactor(i) == 0.0 && diffusionFactor(j) == 0.0) {
		return;
	}
	
	// Large Cluster Reactions
	if (this->_clusterData.enableLargeCluster())
		addSingleSizeReactions(i, j, tag);
	
	// Get the composition of each cluster
	/*const auto& cl1Reg = this->getCluster(i).getRegion();
	const auto& cl2Reg = this->getCluster(j).getRegion();
	Composition lo1 = cl1Reg.getOrigin();
	Composition hi1 = cl1Reg.getUpperLimitPoint();
	Composition lo2 = cl2Reg.getOrigin();
	Composition hi2 = cl2Reg.getUpperLimitPoint();*/

	// vac + vac = vac
	if (lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::V)) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::V] + lo2[Species::V];
		auto hiSize = hi1[Species::V] + hi2[Species::V] - 2;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::V] = size;
			auto vProdId = subpaving.findTileId(comp);
			if (vProdId != subpaving.invalidIndex() &&
				vProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, vProdId});
				if (lo1[Species::V] == 1 || lo2[Species::V] == 1) {
					this->addDissociationReaction(tag, {vProdId, i, j});
				}
				previousIndex = vProdId;

				// Special case to allow size 9 basal clusters to dissociate
				// into vacancies
				if (size == 9 &&
					(lo1[Species::V] == 1 || lo2[Species::V] == 1)) {
					Composition comp = Composition::zero();
					comp[Species::Basal] = size;
					auto basalProdId = subpaving.findTileId(comp);
					if (basalProdId != subpaving.invalidIndex() &&
						basalProdId != previousIndex) {
						// No production (vacancies do not accumulate into basal
						// clusters)
						this->addDissociationReaction(tag, {basalProdId, i, j});
					}
				}
			}
		}

		return;
	}

	// Adding basal
	// Basal + Basal = Basal
	if (lo1.isOnAxis(Species::Basal) && lo2.isOnAxis(Species::Basal)) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::Basal] + lo2[Species::Basal];
		auto hiSize = hi1[Species::Basal] + hi2[Species::Basal] - 2;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::Basal] = size;
			auto vProdId = subpaving.findTileId(comp);
			if (vProdId != subpaving.invalidIndex() &&
				vProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, vProdId});
				if (lo1[Species::Basal] == 1 || lo2[Species::Basal] == 1) {
					// this->addDissociationReaction(tag, {vProdId, i, j});
					// Dissociating basal clusters produce V
				}
				previousIndex = vProdId;
			}
		}

		return;
	}

	// vac + Basal = Basal
	if ((lo1.isOnAxis(Species::Basal) && lo2.isOnAxis(Species::V)) ||
		(lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::Basal))) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::V] + lo2[Species::V] + lo1[Species::Basal] +
			lo2[Species::Basal]; // They can all be added because they are
								 // orthogonal
		auto hiSize = hi1[Species::V] + hi2[Species::V] + hi1[Species::Basal] +
			hi2[Species::Basal] - 4;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			if (size > 9) {
				// Find Basal
				Composition comp = Composition::zero();
				comp[Species::Basal] = size;
				auto basalProdId = subpaving.findTileId(comp);
				if (basalProdId != subpaving.invalidIndex() &&
					basalProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, basalProdId});
					if (lo1[Species::V] == 1 || lo2[Species::V] == 1) {
						this->addDissociationReaction(tag, {basalProdId, i, j});
					}
					previousIndex = basalProdId;
				}
			}
			else {
				// Find V
				Composition comp = Composition::zero();
				comp[Species::V] = size;
				auto vProdId = subpaving.findTileId(comp);
				if (vProdId != subpaving.invalidIndex() &&
					vProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, vProdId});
					previousIndex = vProdId;
				}
			}
		}

		return;
	}

	// int + Basal = Basal | vac | int | recombine
	if ((lo1.isOnAxis(Species::Basal) && lo2.isOnAxis(Species::I)) ||
		(lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::Basal))) {
		// Compute the largest possible product and the smallest one
		int largestProd = (int)hi1[Species::Basal] + (int)hi2[Species::Basal] -
			2 - (int)lo1[Species::I] - (int)lo2[Species::I];
		int smallestProd = (int)lo1[Species::Basal] + (int)lo2[Species::Basal] -
			(int)hi1[Species::I] - (int)hi2[Species::I] + 2;
		// Loop on the products
		for (int prodSize = smallestProd; prodSize <= largestProd; prodSize++) {
			// 4 cases
			if (prodSize > 9) {
				// Looking for Basal cluster
				Composition comp = Composition::zero();
				comp[Species::Basal] = prodSize;
				auto basalProdId = subpaving.findTileId(comp);
				if (basalProdId != subpaving.invalidIndex() &&
					basalProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, basalProdId});
					// No dissociation
					previousIndex = basalProdId;
				}
			}
			else if (prodSize > 0) {
				// Looking for V cluster
				Composition comp = Composition::zero();
				comp[Species::V] = prodSize;
				auto vProdId = subpaving.findTileId(comp);
				if (vProdId != subpaving.invalidIndex() &&
					vProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, vProdId});
					// No dissociation
					previousIndex = vProdId;
				}
			}
			else if (prodSize < 0) {
				// Looking for I cluster
				Composition comp = Composition::zero();
				comp[Species::I] = -prodSize;
				auto iProdId = subpaving.findTileId(comp);
				if (iProdId != subpaving.invalidIndex() &&
					iProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, iProdId});
					// No dissociation
					previousIndex = iProdId;
				}
			}
			else {
				// No product
				this->addProductionReaction(tag, {i, j});
			}
		}

		return;
	}

	// vac + int = vac | int | recombine
	if (((lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::V)) ||
			(lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::I)))) {
		// Compute the largest possible product and the smallest one
		int largestProd = (int)hi1[Species::V] + (int)hi2[Species::V] - 2 -
			(int)lo1[Species::I] - (int)lo2[Species::I];
		int smallestProd = (int)lo1[Species::V] + (int)lo2[Species::V] -
			(int)hi1[Species::I] - (int)hi2[Species::I] + 2;
		// Loop on the products
		for (int prodSize = smallestProd; prodSize <= largestProd; prodSize++) {
			// 3 cases
			if (prodSize > 0) {
				// Looking for V cluster
				Composition comp = Composition::zero();
				comp[Species::V] = prodSize;
				auto vProdId = subpaving.findTileId(comp);
				if (vProdId != subpaving.invalidIndex() &&
					vProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, vProdId});
					// No dissociation
					previousIndex = vProdId;
				}
			}
			else if (prodSize < 0) {
				// Looking for I cluster
				Composition comp = Composition::zero();
				comp[Species::I] = -prodSize;
				auto iProdId = subpaving.findTileId(comp);
				if (iProdId != subpaving.invalidIndex() &&
					iProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, iProdId});
					// No dissociation
					previousIndex = iProdId;
				}
			}
			else {
				// No product
				this->addProductionReaction(tag, {i, j});
			}
		}

		return;
	}

	// int + int = int
	if (lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::I)) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::I] + lo2[Species::I];
		auto hiSize = hi1[Species::I] + hi2[Species::I] - 2;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::I] = size;
			auto iProdId = subpaving.findTileId(comp);
			if (iProdId != subpaving.invalidIndex() &&
				iProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, iProdId});
				if (lo1[Species::I] == 1 || lo2[Species::I] == 1) {
					this->addDissociationReaction(tag, {iProdId, i, j});
				}
				previousIndex = iProdId;
			}
		}

		return;
	}
}

template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
ZrReactionGenerator::addSinks(IndexType i, TTag tag) const
{
	using Species = typename Network::Species;
	using Composition = typename Network::Composition;

	const auto& clReg = this->getCluster(i).getRegion();
	Composition lo = clReg.getOrigin();

	// I
	if (clReg.isSimplex() && lo.isOnAxis(Species::I)) {
		this->addSinkReaction(tag, {i, Network::invalidIndex()});
	}

	// V
	if (clReg.isSimplex() && lo.isOnAxis(Species::V)) {
		this->addSinkReaction(tag, {i, Network::invalidIndex()});
	}
}

inline ReactionCollection<ZrReactionGenerator::Network>
ZrReactionGenerator::getReactionCollection() const
{
	ReactionCollection<Network> ret(this->_clusterData.gridSize,
		this->_clusterData.numClusters, this->_enableReadRates,
		this->getProductionReactions(), this->getDissociationReactions(),
		this->getSinkReactions(), this->getConstantReactions());
	return ret;
}

template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
ZrReactionGenerator::addSingleSizeReactions(
	IndexType i, IndexType j, TTag tag) const
{
	using Species = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;

	auto vacId = this->_clusterData.vacId();
	auto intId = this->_clusterData.intId();
	auto basalId = this->_clusterData.basalId();
	
	auto diffusionFactor = this->_clusterData.diffusionFactor;
	
		// Get the composition of each cluster
	const auto& cl1Reg = this->getCluster(i).getRegion();
	const auto& cl2Reg = this->getCluster(j).getRegion();
	Composition lo1 = cl1Reg.getOrigin();
	Composition hi1 = cl1Reg.getUpperLimitPoint();
	Composition lo2 = cl2Reg.getOrigin();
	Composition hi2 = cl2Reg.getUpperLimitPoint();
/*	if (lo1[Species::V] > 0){
			auto largeSize = std::max(lo1[Species::V], lo2[Species::V]);
			std::cout<<"diffusion factor for v"<<largeSize<<": " << diffusionFactor(i)<<std::endl;
			std::cout<<std::endl;
	}
	if (lo2[Species::V] > 0){
			auto largeSize = std::max(lo1[Species::V], lo2[Species::V]);
			std::cout<<"diffusion factor for v"<<largeSize<<": " << diffusionFactor(j)<<std::endl;
			std::cout<<std::endl;
	}*/
	
	// Check the diffusion factors
	if (diffusionFactor(i) == 0.0 && diffusionFactor(j) == 0.0){
		return;
	}
	
	if (i == j) {
		const auto& clReg = this->getCluster(i).getRegion();
		Composition lo = clReg.getOrigin();

		// Check reaction with largest cluster
		if (not clReg.isSimplex())
			return;

		// V case
		if (lo.isOnAxis(Species::V)) {
			// V_k + L -> L
			this->addProductionReaction(tag, {i, vacId, vacId});
		}
		// I case
		else if (lo.isOnAxis(Species::I)) {
			// I_k + L -> L
			this->addProductionReaction(tag, {i, vacId, vacId});
		}
	}

	

	// Find the edge of the phase space
	const auto& largestReg = this->getCluster(largestClusterId).getRegion();
	Composition hiLargest = largestReg.getUpperLimitPoint();
	auto largestSize = hiLargest[Species::V] - 1; // Don't know which one was saved
	

	// V_a + V_b -> V
	if (hi1[Species::V] + hi2[Species::V] - 2 > largestSize) {
		this->addProductionReaction(tag, {i, j, vacId});
	}

	// I_a + V -> V_b
	if ((lo1.isOnAxis(Species::I) and lo2.isOnAxis(Species::V)) or
		(lo1.isOnAxis(Species::V) and lo2.isOnAxis(Species::I))) {
		// It should be around the largest size value
		if (hi1[Species::V] + hi2[Species::V] + hi1[Species::I] +
				hi2[Species::I] - 4 > largestSize) {
			// Need to know which one is I
			auto iId = lo1[Species::I] > 0 ? i : j;
			auto vId = lo1[Species::I] > 0 ? j : i;
			this->addProductionReaction(tag, {iId, vacId, vId});
		}
	}
}

} // namespace detail

inline detail::ZrReactionGenerator
ZrReactionNetwork::getReactionGenerator() const noexcept
{
	return detail::ZrReactionGenerator{*this};
}

namespace detail
{
KOKKOS_INLINE_FUNCTION
void
ZrClusterUpdater::updateDiffusionCoefficient(
	const ClusterData& data, IndexType clusterId, IndexType gridIndex) const
{
	// I migration energies in eV
	constexpr Kokkos::Array<double, 6> iMigrationA = {
		0.0, 0.17, 0.23, 0.49, 0.75, 0.87};
	constexpr Kokkos::Array<double, 6> iMigrationC = {
		0.0, 0.30, 0.54, 0.93, 1.2, 1.6};
	// I diffusion factors in nm^2/s
	constexpr Kokkos::Array<double, 6> iDiffusionA = {
		0.0, 2.4e+11, 3.2e+11, 4.9e+12, 5.1e+13, 4.3e+13};
	constexpr Kokkos::Array<double, 6> iDiffusionC = {
		0.0, 6.8e+11, 2.6e+12, 6.8e+13, 4.2e+14, 5.5e+15};

	// V migration energies in eV (up to n = 6)
	constexpr Kokkos::Array<double, 7> vMigrationA = {
		0.0, 0.59, 0.58, 0.94, 0.16, 0.81, 0.25};
	constexpr Kokkos::Array<double, 7> vMigrationC = {
		0.0, 0.67, 0.41, 1.12, 0.58, 0.29, 0.18};
	// V diffusions factors in nm^2/s
	constexpr Kokkos::Array<double, 7> vDiffusionA = {
		0.0, 1.6e+12, 2.7e+12, 4.9e+13, 2.5e+10, 2e+13, 3.2e+10};
	constexpr Kokkos::Array<double, 7> vDiffusionC = {
		0.0, 2.2e+12, 2.3e+11, 1.27e+15, 4.5e+11, 5.7e+11, 9.1e+9};

	// 3D diffuser case
	if (data.migrationEnergy(clusterId) < 0.0) {
		double kernel = -1.0 / (kBoltzmann * data.temperature(gridIndex));
		const auto& clReg = data.getCluster(clusterId).getRegion();
		Network::Composition lo = clReg.getOrigin();
		using Species = Network::Species;

		if (lo.isOnAxis(Species::I)) {
			// Compute each contribution
			double Da = iDiffusionA[lo[Species::I]] *
				exp(iMigrationA[lo[Species::I]] * kernel);
			double Dc = iDiffusionC[lo[Species::I]] *
				exp(iMigrationC[lo[Species::I]] * kernel);

			// Compute the mean
			data.diffusionCoefficient(clusterId, gridIndex) =
				pow(Da * Da * Dc, 1.0 / 3.0);

			// Compute the anisotropy factor
			data.extraData.anisotropyRatio(clusterId, gridIndex) =
				pow(Dc / Da, 1.0 / 6.0);
			
			return;
		}

		if (lo.isOnAxis(Species::V)) {
			// Compute each contribution
			double Da = vDiffusionA[lo[Species::V]] *
				exp(vMigrationA[lo[Species::V]] * kernel);
			double Dc = vDiffusionC[lo[Species::V]] *
				exp(vMigrationC[lo[Species::V]] * kernel);

			// Compute the mean
			data.diffusionCoefficient(clusterId, gridIndex) =
				pow(Da * Da * Dc, 1.0 / 3.0);

			// Compute the anisotropy factor
			data.extraData.anisotropyRatio(clusterId, gridIndex) =
				pow(Dc / Da, 1.0 / 6.0);
			
			return;
		}
	}

	// 1D diffuser case
	data.diffusionCoefficient(clusterId, gridIndex) =
		data.diffusionFactor(clusterId) *
		exp(-data.migrationEnergy(clusterId) /
			(kBoltzmann * data.temperature(gridIndex)));
}
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
