#include "HDF5NetworkLoader.h"
#include <stdio.h>
#include <limits>
#include <algorithm>
#include <vector>
#include "PSIClusterReactionNetwork.h"
#include <xolotlPerf.h>
#include "xolotlCore/io/XFile.h"

namespace xolotlCore {

std::unique_ptr<IReactionNetwork> HDF5NetworkLoader::load(
		const IOptions& options) {
	// Get the dataset from the HDF5 files
	int normalSize = 0, superSize = 0;
	XFile networkFile(fileName);
	auto networkGroup = networkFile.getGroup<XFile::NetworkGroup>();
	assert(networkGroup);
	auto list = networkGroup->readNetworkSize(normalSize, superSize);

	// Initialization
	int numHe = 0, numV = 0, numI = 0, numD = 0, numT = 0;
	double formationEnergy = 0.0, migrationEnergy = 0.0;
	double diffusionFactor = 0.0;
	std::vector<std::reference_wrapper<Reactant> > reactants;

	// Prepare the network
	std::unique_ptr<PSIClusterReactionNetwork> network(
			new PSIClusterReactionNetwork(handlerRegistry));

	// Set the lattice parameter in the network
	double latticeParam = options.getLatticeParameter();
	if (!(latticeParam > 0.0))
		latticeParam = tungstenLatticeConstant;
	network->setLatticeParameter(latticeParam);

	// Set the helium radius in the network
	double radius = options.getImpurityRadius();
	if (!(radius > 0.0))
		radius = heliumRadius;
	network->setImpurityRadius(radius);

	// Set the interstitial bias in the network
	network->setInterstitialBias(options.getBiasFactor());

	// Set the hydrogan radius factor
	hydrogenRadiusFactor = options.getHydrogenFactor();

	// Loop on the clusters
	for (int i = 0; i < normalSize + superSize; i++) {
		// Open the cluster group
		XFile::ClusterGroup clusterGroup(*networkGroup, i);

		if (i < normalSize) {
			// Normal cluster
			// Read the composition
			auto comp = clusterGroup.readCluster(formationEnergy,
					migrationEnergy, diffusionFactor);
			numHe = comp[toCompIdx(Species::He)];
			numD = comp[toCompIdx(Species::D)];
			numT = comp[toCompIdx(Species::T)];
			numV = comp[toCompIdx(Species::V)];
			numI = comp[toCompIdx(Species::I)];

			// Create the cluster
			auto nextCluster = createPSICluster(numHe, numD, numT, numV, numI,
					*network);

			// Set the formation energy
			nextCluster->setFormationEnergy(formationEnergy);
			// Set the diffusion factor and migration energy
			nextCluster->setMigrationEnergy(migrationEnergy);
			nextCluster->setDiffusionFactor(diffusionFactor);

			// Save it in the network
			pushPSICluster(network, reactants, nextCluster);
		} else {
			// Super cluster
			auto list = clusterGroup.readPSISuperCluster();

			// Create the cluster
			auto nextCluster = createPSISuperCluster(list, *network);

			// Save it in the network
			pushPSICluster(network, reactants, nextCluster);
		}
	}

	// Ask reactants to update now that they are in network.
	for (IReactant& currReactant : reactants) {
		currReactant.updateFromNetwork();
	}

	// Define the phase space for the network
	int nDim = 1;
	// Loop on the list for the phase space
	for (int i = 1; i < 5; i++)
		if (list[i] > 0)
			nDim++;

	// Now that all the clusters are created
	// Give the information on the phase space to the network
	network->setPhaseSpace(nDim, list);

	// Set the reactions
	networkGroup->readReactions(*network);

	// Recompute Ids and network size
	network->reinitializeNetwork();

	// Need to use move() because return type uses smart pointer to base class,
	// not derived class that we created.
	// Some C++11 compilers accept it without the move, but apparently
	// that is not correct behavior until C++14.
	return std::move(network);
}

} // namespace xolotlCore

