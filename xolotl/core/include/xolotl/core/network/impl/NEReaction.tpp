#pragma once

#include <xolotl/core/network/impl/NucleationReaction.tpp>
#include <xolotl/core/network/impl/ReSolutionReaction.tpp>
#include <xolotl/core/network/impl/Reaction.tpp>

namespace xolotl
{
namespace core
{
namespace network
{
KOKKOS_INLINE_FUNCTION
double
NEDissociationReaction::computeBindingEnergy()
{
	auto cl = this->_clusterData->getCluster(this->_reactant);
	auto prod1 = this->_clusterData->getCluster(this->_products[0]);
	auto prod2 = this->_clusterData->getCluster(this->_products[1]);
	return prod1.getFormationEnergy() + prod2.getFormationEnergy() -
		cl.getFormationEnergy();
}
} // namespace network
} // namespace core
} // namespace xolotl
