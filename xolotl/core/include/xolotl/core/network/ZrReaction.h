#pragma once

#include <xolotl/core/network/ConstantReaction.h>
#include <xolotl/core/network/SinkReaction.h>
#include <xolotl/core/network/ZrTraits.h>

namespace xolotl
{
namespace core
{
namespace network
{
class ZrReactionNetwork;

class ZrProductionReaction :
	public ProductionReaction<ZrReactionNetwork, ZrProductionReaction>
{
public:
	using Superclass =
		ProductionReaction<ZrReactionNetwork, ZrProductionReaction>;

	using Superclass::Superclass;
	using NetworkType = typename Superclass::NetworkType;
	using ReactionDataRef = typename Superclass::ReactionDataRef;
	using ClusterData = typename Superclass::ClusterData;
	using IndexType = typename Superclass::IndexType;
	using Composition = typename Superclass::Composition;
	using Region = typename Superclass::Region;
	using ConcentrationsView = typename Superclass::ConcentrationsView;
	using FluxesView = typename Superclass::FluxesView;
	using Species = typename Superclass::Species;
	
	KOKKOS_INLINE_FUNCTION
	ZrProductionReaction(ReactionDataRef reactionData,
		const ClusterData& clusterData, IndexType reactionId,
		IndexType cluster0, IndexType cluster1,
		IndexType cluster2 = Superclass::invalidIndex,
		IndexType cluster3 = Superclass::invalidIndex)
		
	{
		this->_clusterData = &clusterData;
		this->_reactionId = reactionId;
		this->_rate = reactionData.getRates(reactionId);
		this->_widths = reactionData.getWidths(reactionId);
		this->_coefs = reactionData.getCoefficients(reactionId);

		this->_reactants = {cluster0, cluster1};
		this->_products = {cluster2, cluster3};
		
		auto numClusters = clusterData.numClusters;
		// Check if the single size is involved
		if (cluster0 >= numClusters)
			isLargeClusterReaction = true;
		if (cluster1 >= numClusters)
			isLargeClusterReaction = true;
		if (cluster2 != Superclass::invalidIndex and cluster2 >= numClusters)
			isLargeClusterReaction = true;
		if (cluster3 != Superclass::invalidIndex and cluster3 >= numClusters)
			isLargeClusterReaction = true;
		
		//static
		const auto dummyRegion = Region(Composition{});
		
		for (auto i : {0,1}){
			if (this->_reactants[i] < numClusters){
				this-> copyMomentIds(this->_reactants[i], this->_reactantMomentIds[i]);
			}
			else{
				auto shift = (this->_reactants[i] - numClusters)/2;
				switch (shift) {
				// Vac
				case 0:
					this->_reactantMomentIds[i][0] =
						this->_clusterData->vacAvId();
					break;
				// Int
				case 1:
					this->_reactantMomentIds[i][0] =
						this->_clusterData->intAvId();
					break;
				// Basal
				case 2:
					this->_reactantMomentIds[i][0] =
						this->_clusterData->basalAvId();
					break;
				}
			}
			if (this->_products[i] < numClusters){
				this->copyMomentIds(this->_products[i], this->_productMomentIds[i]);
			}
			else {
				if (this->_products[i] == Superclass::invalidIndex) {
					for (IndexType j = 0; j < Superclass::nMomentIds; ++j) {
						this->_productMomentIds[i][j] = Superclass::invalidIndex;
					}
				}
				else {
					auto shift = (this->_products[i] - numClusters)/2;
					switch (shift) {
					// Vac
					case 0:
						this->_productMomentIds[i][0] =
							this->_clusterData->vacAvId();
						break;
					// Int
					case 1:
						this->_productMomentIds[i][0] =
							this->_clusterData->intAvId();
						break;
					// Basal
					case 2:
						this->_productMomentIds[i][0] =
							this->_clusterData->basalAvId();
						break;
					}
				}
			}
		}
		
		const auto& cl1Reg = (this->_reactants[0] < numClusters) ?
			this->_clusterData->getCluster(this->_reactants[0]).getRegion() :
			dummyRegion;
		const auto& cl2Reg = (this->_reactants[1] < numClusters) ?
			this->_clusterData->getCluster(this->_reactants[1]).getRegion() :
			dummyRegion;
		const auto& pr1Reg = (this->_products[0] == Superclass::invalidIndex) ?
			dummyRegion :
			(this->_products[0] < numClusters) ?
			this->_clusterData->getCluster(this->_products[0]).getRegion() :
			dummyRegion;
		const auto& pr2Reg = (this->_products[1] == Superclass::invalidIndex) ?
			dummyRegion :
			(this->_products[1] < numClusters) ?
			this->_clusterData->getCluster(this->_products[1]).getRegion() :
			dummyRegion;

		this->_reactantVolumes = {cl1Reg.volume(), cl2Reg.volume()};
		this->_productVolumes = {pr1Reg.volume(), pr2Reg.volume()};

		this->initialize();
	}
	
	KOKKOS_INLINE_FUNCTION
	ZrProductionReaction(ReactionDataRef reactionData,
		const ClusterData& clusterData, IndexType reactionId,
		const detail::ClusterSet& clusterSet) :
		ZrProductionReaction(reactionData, clusterData, reactionId,
			clusterSet.cluster0, clusterSet.cluster1, clusterSet.cluster2,
			clusterSet.cluster3)
	{
	}
		
	KOKKOS_INLINE_FUNCTION
	double
	getRateForProduction(IndexType gridIndex);
	
	KOKKOS_INLINE_FUNCTION
	void
	computeCoefficients();

	KOKKOS_INLINE_FUNCTION
	void
	computeFlux(ConcentrationsView concentrations, FluxesView fluxes,
		IndexType gridIndex);

	KOKKOS_INLINE_FUNCTION
	void
	computePartialDerivatives(ConcentrationsView concentrations,
		Kokkos::View<double*> values, IndexType gridIndex);

private:
	bool isLargeClusterReaction = false;
};

class ZrDissociationReaction :
	public DissociationReaction<ZrReactionNetwork, ZrDissociationReaction>
{
public:
	using Superclass =
		DissociationReaction<ZrReactionNetwork, ZrDissociationReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	getRateForProduction(IndexType gridIndex);

	KOKKOS_INLINE_FUNCTION
	double
	computeBindingEnergy(double time = 0.0);
};

class ZrSinkReaction : public SinkReaction<ZrReactionNetwork, ZrSinkReaction>
{
public:
	using Superclass = SinkReaction<ZrReactionNetwork, ZrSinkReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	computeRate(IndexType gridIndex, double time = 0.0);
};

class ZrConstantReaction :
	public ConstantReaction<ZrReactionNetwork, ZrConstantReaction>
{
public:
	using Superclass = ConstantReaction<ZrReactionNetwork, ZrConstantReaction>;

	using Superclass::Superclass;
};
} // namespace network
} // namespace core
} // namespace xolotl
