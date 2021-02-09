#pragma once

#include <xolotl/core/network/NETraits.h>
#include <xolotl/core/network/NucleationReaction.h>
#include <xolotl/core/network/ReSolutionReaction.h>
#include <xolotl/core/network/SinkReaction.h>

namespace xolotl
{
namespace core
{
namespace network
{
class NEReactionNetwork;

class NEProductionReaction :
	public ProductionReaction<NEReactionNetwork, NEProductionReaction>
{
public:
	using Superclass =
		ProductionReaction<NEReactionNetwork, NEProductionReaction>;

	using Superclass::Superclass;
};

class NEDissociationReaction :
	public DissociationReaction<NEReactionNetwork, NEDissociationReaction>
{
public:
	using Superclass =
		DissociationReaction<NEReactionNetwork, NEDissociationReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	computeBindingEnergy();
};

class NEReSolutionReaction :
	public ReSolutionReaction<NEReactionNetwork, NEReSolutionReaction>
{
public:
	using Superclass =
		ReSolutionReaction<NEReactionNetwork, NEReSolutionReaction>;

	using Superclass::Superclass;
};

class NENucleationReaction :
	public NucleationReaction<NEReactionNetwork, NENucleationReaction>
{
public:
	using Superclass =
		NucleationReaction<NEReactionNetwork, NENucleationReaction>;

	using Superclass::Superclass;
};

class NESinkReaction : public SinkReaction<NEReactionNetwork, NESinkReaction>
{
public:
	using Superclass = SinkReaction<NEReactionNetwork, NESinkReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	getSinkBias();
};
} // namespace network
} // namespace core
} // namespace xolotl
