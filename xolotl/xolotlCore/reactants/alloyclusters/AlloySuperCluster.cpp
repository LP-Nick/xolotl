// Includes
#include "AlloySuperCluster.h"
#include <Constants.h>
#include <MathUtils.h>
#include <AlloyClusterReactionNetwork.h>

using namespace xolotlCore;

/**
 * The moment partials.
 */
std::vector<double> momentPartials;

AlloySuperCluster::AlloySuperCluster(int numMax, int nTot,
		ReactantType typeName, IReactionNetwork& _network,
		std::shared_ptr<xolotlPerf::IHandlerRegistry> registry) :
		AlloyCluster(_network, registry), nTot(nTot), l0(0.0), l1(0.0), momentFlux(
				0.0) {
	// Initialize the dispersion sum
	double nAtomSquare = 0.0;
	reactionRadius = 0.0, formationEnergy = 0.0;
	// Loop on the contained size
	for (int i = numMax; i > numMax - nTot; i--) {
		size += i;
		reactionRadius += _network.getReactionRadius(typeName, i);
		formationEnergy += _network.getFormationEnergy(typeName, i);
		nAtomSquare += (double) i * i;
	}

	// Set the cluster sizes
	numAtom = (double) size / (double) nTot;
	size = (int) numAtom;
	reactionRadius = reactionRadius / (double) nTot;
	formationEnergy = formationEnergy / (double) nTot;

	// Compute the dispersion
	if (nTot == 1)
		dispersion = 1.0;
	else {
		dispersion = 2.0 * (nAtomSquare - (numAtom * (double) nTot * numAtom))
				/ ((double) (nTot * (nTot - 1)));
	}

	// Update the composition map
	composition[toCompIdx(toSpecies(typeName))] = size;

	// Set the diffusion factor and the migration energy
	migrationEnergy = std::numeric_limits<double>::infinity();
	diffusionFactor = 0.0;

	// Set the typename appropriately
	if (typeName == ReactantType::Void)
		type = ReactantType::VoidSuper;
	else if (typeName == ReactantType::Perfect)
		type = ReactantType::PerfectSuper;
	else if (typeName == ReactantType::Frank)
		type = ReactantType::FrankSuper;
	else if (typeName == ReactantType::Faulted)
		type = ReactantType::FaultedSuper;

	// Set the reactant name appropriately
	std::stringstream nameStream;
	nameStream << toString(type) << "_" << numAtom;
	name = nameStream.str();

	return;
}

void AlloySuperCluster::resultFrom(ProductionReaction& reaction,
		IReactant& product) {
	// Cast the reacting clusters
	auto& cluster1 = static_cast<AlloyCluster&>(reaction.first);
	auto& cluster2 = static_cast<AlloyCluster&>(reaction.second);
	auto& prodCluster = static_cast<AlloyCluster&>(product);

	// Compute the overlap
	auto & alloyNetwork = static_cast<AlloyClusterReactionNetwork&>(network);
	int width1 = cluster1.getSectionWidth();
	int size1 = cluster1.getSize();
	int width2 = cluster2.getSectionWidth();
	int size2 = cluster2.getSize();
	int prodWidth = prodCluster.getSectionWidth(), prodSize =
			prodCluster.getSize();
	int lo1 = alloyNetwork.typeSwitch(cluster1.getType())
			* alloyNetwork.typeSwitch(prodCluster.getType())
			* ((int) ((double) size1 - (double) width1 / 2.0) + 1), lo2 =
			alloyNetwork.typeSwitch(cluster2.getType())
					* alloyNetwork.typeSwitch(prodCluster.getType())
					* ((int) ((double) size2 - (double) width2 / 2.0) + 1),
			hi1 = alloyNetwork.typeSwitch(cluster1.getType())
					* alloyNetwork.typeSwitch(prodCluster.getType())
					* ((int) ((double) size1 + (double) width1 / 2.0)), hi2 =
					alloyNetwork.typeSwitch(cluster2.getType())
							* alloyNetwork.typeSwitch(prodCluster.getType())
							* ((int) ((double) size2 + (double) width2 / 2.0));
	int prodLo = ((int) ((double) prodSize - (double) prodWidth / 2.0) + 1),
			prodHi = ((int) ((double) prodSize + (double) prodWidth / 2.0));

	int overlap = std::min(prodHi, hi1 + hi2) - std::max(prodLo, lo1 + lo2) + 1;

	// Skip if the reaction doesn't overlap
	if (overlap < 1)
		return;

	// Create a new SuperClusterProductionPair
	SuperClusterProductionPair superPair(&cluster1, &cluster2, &reaction);
	// Compute the coefficients
	superPair.a000 = overlap;
	superPair.a001 = firstOrderSum(std::max(prodLo, lo1 + lo2),
			std::min(prodHi, hi1 + hi2), numAtom) / dispersion;
	if (width1 > 1) {
		superPair.a100 = 2.0
				* firstOrderSum(std::max(prodLo - lo2, lo1),
						std::min(prodHi - hi2, hi1), (double) (lo1 + hi1) / 2.0)
				/ (double) (hi1 - lo1);
		superPair.a101 = 2.0
				* secondOrderOffsetSum(std::max(prodLo - lo2, lo1),
						std::min(prodHi - hi2, hi1), (double) (lo1 + hi1) / 2.0,
						numAtom, lo2) / (dispersion * (double) (hi1 - lo1));
	} else {
		superPair.a100 = 0.0;
		superPair.a101 = 0.0;
	}
	if (width2 > 1) {
		superPair.a010 = 2.0
				* firstOrderSum(std::max(prodLo - lo1, lo2),
						std::min(prodHi - hi1, hi2), (double) (lo2 + hi2) / 2.0)
				/ (double) ((hi2 - lo2));
		superPair.a011 = 2.0
				* secondOrderOffsetSum(std::max(prodLo - lo1, lo2),
						std::min(prodHi - hi1, hi2), (double) (lo2 + hi2) / 2.0,
						numAtom, lo1) / (dispersion * (double) (hi2 - lo2));
	} else {
		superPair.a010 = 0.0;
		superPair.a011 = 0.0;
	}
	if (width1 > 1 && width2 > 1) {
		// Should never happen for now
		std::cout << "Both reactants are super: " << cluster1.getName() << " + "
				<< cluster2.getName() << " -> " << name << std::endl;
	} else {
		superPair.a110 = 0.0;
		superPair.a111 = 0.0;
	}
	// Add it to the list
	effReactingList.push_front(superPair);

	return;
}

void AlloySuperCluster::resultFrom(ProductionReaction& reaction, double *coef) {

	// Create a new SuperClusterProductionPair
	SuperClusterProductionPair superPair(
			&static_cast<AlloyCluster&>(reaction.first),
			&static_cast<AlloyCluster&>(reaction.second), &reaction);
	// Update the coeficients
	superPair.a000 = coef[0];
	superPair.a001 = coef[1];
	superPair.a100 = coef[2];
	superPair.a101 = coef[3];
	superPair.a010 = coef[4];
	superPair.a011 = coef[5];
	superPair.a110 = coef[6];
	superPair.a111 = coef[7];

	// Add it to the list
	effReactingList.push_front(superPair);

	return;
}

void AlloySuperCluster::participateIn(ProductionReaction& reaction,
		IReactant& product) {
	// Look for the other cluster
	auto& otherCluster = static_cast<AlloyCluster&>(
			(reaction.first.getId() == id) ? reaction.second : reaction.first);
	auto& prodCluster = static_cast<AlloyCluster&>(product);

	// Compute the overlap
	auto & alloyNetwork = static_cast<AlloyClusterReactionNetwork&>(network);
	int width1 = otherCluster.getSectionWidth();
	int size1 = otherCluster.getSize();
	int width2 = getSectionWidth();
	int size2 = getSize();
	int prodWidth = prodCluster.getSectionWidth(), prodSize =
			prodCluster.getSize();
	int lo1 = alloyNetwork.typeSwitch(otherCluster.getType())
			* alloyNetwork.typeSwitch(prodCluster.getType())
			* ((int) ((double) size1 - (double) width1 / 2.0) + 1), lo2 =
			alloyNetwork.typeSwitch(getType())
					* alloyNetwork.typeSwitch(prodCluster.getType())
					* ((int) ((double) size2 - (double) width2 / 2.0) + 1),
			hi1 = alloyNetwork.typeSwitch(otherCluster.getType())
					* alloyNetwork.typeSwitch(prodCluster.getType())
					* ((int) ((double) size1 + (double) width1 / 2.0)), hi2 =
					alloyNetwork.typeSwitch(getType())
							* alloyNetwork.typeSwitch(prodCluster.getType())
							* ((int) ((double) size2 + (double) width2 / 2.0));
	int prodLo = ((int) ((double) prodSize - (double) prodWidth / 2.0) + 1),
			prodHi = ((int) ((double) prodSize + (double) prodWidth / 2.0));

	int overlap = std::min(prodHi, hi1 + hi2) - std::max(prodLo, lo1 + lo2) + 1;

	// Skip if the reaction doesn't overlap
	if (overlap < 1)
		return;

	// Create a new SuperClusterProductionPair
	SuperClusterProductionPair superPair(&otherCluster, nullptr, &reaction);
	// Compute the coefficients
	superPair.a000 = overlap;
	superPair.a001 = firstOrderSum(std::max(prodLo - lo1, lo2),
			std::min(prodHi - hi1, hi2), numAtom) / dispersion;
	if (width1 > 1) {
		// Should never happen for now
		std::cout << "Combining reactant is super: " << otherCluster.getName()
				<< " + " << name << std::endl;
	} else {
		superPair.a010 = 0.0;
		superPair.a011 = 0.0;
		superPair.a110 = 0.0;
		superPair.a111 = 0.0;
	}

	if (width2 > 1) {
		superPair.a100 = 2.0
				* firstOrderSum(std::max(prodLo - lo1, lo2),
						std::min(prodHi - hi1, hi2), numAtom)
				/ (double) ((hi2 - lo2));
		superPair.a101 = 2.0
				* secondOrderSum(std::max(prodLo - lo1, lo2),
						std::min(prodHi - hi1, hi2), numAtom)
				/ (dispersion * (double) (hi2 - lo2));

	} else {
		superPair.a100 = 0.0;
		superPair.a101 = 0.0;
	}

	// Add it to the list
	effCombiningList.push_front(superPair);

	return;
}

void AlloySuperCluster::participateIn(ProductionReaction& reaction,
		double *coef) {
	// Look for the other cluster
	auto& otherCluster = static_cast<AlloyCluster&>(
			(reaction.first.getId() == id) ? reaction.second : reaction.first);

	// Create a new SuperClusterProductionPair with NULL as the second cluster because
	// we do not need it
	SuperClusterProductionPair superPair(&otherCluster, nullptr, &reaction);
	// Update the coeficients
	superPair.a000 = coef[0];
	superPair.a001 = coef[1];
	superPair.a100 = coef[2];
	superPair.a101 = coef[3];
	superPair.a010 = coef[4];
	superPair.a011 = coef[5];
	superPair.a110 = coef[6];
	superPair.a111 = coef[7];

	// Add it to the list
	effCombiningList.push_front(superPair);

	return;
}

void AlloySuperCluster::participateIn(DissociationReaction& reaction,
		IReactant& disso) {
	// Look for the other cluster
	auto& emittedCluster = static_cast<AlloyCluster&>(
			(reaction.first.getId() == id) ? reaction.second : reaction.first);
	auto& dissoCluster = static_cast<AlloyCluster&>(disso);

	// Compute the overlap
	auto & alloyNetwork = static_cast<AlloyClusterReactionNetwork&>(network);
	int width1 = emittedCluster.getSectionWidth();
	int size1 = emittedCluster.getSize();
	int width2 = getSectionWidth();
	int size2 = getSize();
	int dissoWidth = dissoCluster.getSectionWidth(), dissoSize =
			dissoCluster.getSize();
	int lo1 = alloyNetwork.typeSwitch(emittedCluster.getType())
			* alloyNetwork.typeSwitch(dissoCluster.getType())
			* ((int) ((double) size1 - (double) width1 / 2.0) + 1), lo2 =
			alloyNetwork.typeSwitch(getType())
					* alloyNetwork.typeSwitch(dissoCluster.getType())
					* ((int) ((double) size2 - (double) width2 / 2.0) + 1),
			hi1 = alloyNetwork.typeSwitch(emittedCluster.getType())
					* alloyNetwork.typeSwitch(dissoCluster.getType())
					* ((int) ((double) size1 + (double) width1 / 2.0)), hi2 =
					alloyNetwork.typeSwitch(getType())
							* alloyNetwork.typeSwitch(dissoCluster.getType())
							* ((int) ((double) size2 + (double) width2 / 2.0));
	int dissoLo = ((int) ((double) dissoSize - (double) dissoWidth / 2.0) + 1),
			dissoHi = ((int) ((double) dissoSize + (double) dissoWidth / 2.0));

	int overlap = std::min(dissoHi, hi1 + hi2) - std::max(dissoLo, lo1 + lo2)
			+ 1;

	// Skip if the reaction doesn't overlap
	if (overlap < 1)
		return;

	// Create a new SuperClusterProductionPair
	SuperClusterDissociationPair superPair(&dissoCluster, &emittedCluster,
			&reaction);
	// Compute the coefficients
	superPair.a00 = overlap;
	superPair.a01 = firstOrderSum(std::max(dissoLo - lo1, lo2),
			std::min(dissoHi - hi1, hi2), numAtom) / dispersion;
	if (dissoWidth > 1) {
		superPair.a10 = 2.0
				* firstOrderSum(std::max(dissoLo, lo1 + lo2),
						std::min(dissoHi, hi1 + hi2),
						(double) (dissoLo + dissoHi) / 2.0)
				/ (double) (dissoHi - dissoLo);
		superPair.a11 = 2.0
				* secondOrderOffsetSum(std::max(dissoLo, lo1 + lo2),
						std::min(dissoHi, hi1 + hi2),
						(double) (dissoLo + dissoHi) / 2.0, numAtom, lo1)
				/ (dispersion * (double) (dissoHi - dissoLo));
	} else {
		superPair.a10 = 0.0;
		superPair.a11 = 0.0;
	}

	// Add it to the list
	effDissociatingList.push_front(superPair);

	return;
}

void AlloySuperCluster::participateIn(DissociationReaction& reaction,
		double *coef) {
	// Look for the other cluster
	auto& emittedCluster = static_cast<AlloyCluster&>(
			(reaction.first.getId() == id) ? reaction.second : reaction.first);

	// Create a new SuperClusterDissociationPair
	SuperClusterDissociationPair superPair(
			&static_cast<AlloyCluster&>(reaction.dissociating), &emittedCluster,
			&reaction);
	// Update the coeficients
	superPair.a00 = coef[0];
	superPair.a01 = coef[1];
	superPair.a10 = coef[2];
	superPair.a11 = coef[3];

	// Add it to the list
	effDissociatingList.push_front(superPair);

	return;
}

void AlloySuperCluster::emitFrom(DissociationReaction& reaction,
		IReactant& disso) {
	// Cast the reacting clusters
	auto& cluster1 = static_cast<AlloyCluster&>(reaction.first);
	auto& cluster2 = static_cast<AlloyCluster&>(reaction.second);
	auto& dissoCluster = static_cast<AlloyCluster&>(disso);

	// Compute the overlap
	auto & alloyNetwork = static_cast<AlloyClusterReactionNetwork&>(network);
	int width1 = cluster1.getSectionWidth();
	int size1 = cluster1.getSize();
	int width2 = cluster2.getSectionWidth();
	int size2 = cluster2.getSize();
	int dissoWidth = dissoCluster.getSectionWidth(), dissoSize =
			dissoCluster.getSize();
	int lo1 = alloyNetwork.typeSwitch(cluster1.getType())
			* alloyNetwork.typeSwitch(dissoCluster.getType())
			* ((int) ((double) size1 - (double) width1 / 2.0) + 1), lo2 =
			alloyNetwork.typeSwitch(cluster2.getType())
					* alloyNetwork.typeSwitch(dissoCluster.getType())
					* ((int) ((double) size2 - (double) width2 / 2.0) + 1),
			hi1 = alloyNetwork.typeSwitch(cluster1.getType())
					* alloyNetwork.typeSwitch(dissoCluster.getType())
					* ((int) ((double) size1 + (double) width1 / 2.0)), hi2 =
					alloyNetwork.typeSwitch(cluster2.getType())
							* alloyNetwork.typeSwitch(dissoCluster.getType())
							* ((int) ((double) size2 + (double) width2 / 2.0));
	int dissoLo = ((int) ((double) dissoSize - (double) dissoWidth / 2.0) + 1),
			dissoHi = ((int) ((double) dissoSize + (double) dissoWidth / 2.0));

	int overlap = std::min(dissoHi, hi1 + hi2) - std::max(dissoLo, lo1 + lo2)
			+ 1;

	// Skip if the reaction doesn't overlap
	if (overlap < 1)
		return;

	// Create a new SuperClusterProductionPair
	SuperClusterDissociationPair superPair(&cluster1, &cluster2, &reaction);
	// Compute the coefficients
	superPair.a00 = overlap;
	superPair.a01 = firstOrderSum(std::max(dissoLo, lo1 + lo2),
			std::min(dissoHi, hi1 + hi2), numAtom) / dispersion;
	if (dissoWidth > 1) {
		superPair.a10 = 2.0
				* firstOrderSum(std::max(dissoLo, lo1 + lo2),
						std::min(dissoHi, hi1 + hi2), numAtom)
				/ (double) (dissoHi - dissoLo);
		superPair.a11 = 2.0
				* secondOrderSum(std::max(dissoLo, lo1 + lo2),
						std::min(dissoHi, hi1 + hi2), numAtom)
				/ (dispersion * (double) (dissoHi - dissoLo));
	} else {
		superPair.a10 = 0.0;
		superPair.a11 = 0.0;
	}

	// Add it to the list
	effEmissionList.push_front(superPair);

	return;
}

void AlloySuperCluster::emitFrom(DissociationReaction& reaction, double *coef) {

	// Create a new SuperClusterDissociationPair
	SuperClusterDissociationPair superPair(
			&static_cast<AlloyCluster&>(reaction.first),
			&static_cast<AlloyCluster&>(reaction.second), &reaction);
	// Update the coeficients
	superPair.a00 = coef[0];
	superPair.a01 = coef[1];
	superPair.a10 = coef[2];
	superPair.a11 = coef[3];

	// Add it to the list
	effEmissionList.push_front(superPair);

	return;
}

double AlloySuperCluster::getConcentration(double distAtom,
		double distB) const {
	return l0 + (distAtom * l1);
}

double AlloySuperCluster::getMoment() const {
	return l1;
}

double AlloySuperCluster::getTotalConcentration() const {
	// Initial declarations
	int index = 0;
	double distance = 0.0, conc = 0.0;

	// Loop on the xenon width
	for (int k = 0; k < nTot; k++) {
		// Compute the xenon index
		index = (int) (numAtom - (double) nTot / 2.0) + k + 1;

		// Compute the distances
		distance = getDistance(index);

		// Add the concentration of each cluster in the group
		conc += getConcentration(distance);
	}

	return conc;
}

double AlloySuperCluster::getTotalAtomConcentration() const {
	// Initial declarations
	int index = 0;
	double distance = 0.0, conc = 0.0;

	// Loop on the xenon width
	for (int k = 0; k < nTot; k++) {
		// Compute the xenon index
		index = (int) (numAtom - (double) nTot / 2.0) + k + 1;

		// Compute the distances
		distance = getDistance(index);

		// Add the concentration of each cluster in the group times the number of xenon atoms
		conc += getConcentration(distance) * (double) index;
	}

	return conc;
}

double AlloySuperCluster::getDistance(int atom) const {
	if (nTot == 1)
		return 0.0;
	return 2.0 * (double) (atom - numAtom) / ((double) nTot - 1.0);
}

void AlloySuperCluster::resetConnectivities() {
	// Clear both sets
	reactionConnectivitySet.clear();
	dissociationConnectivitySet.clear();

	// Connect this cluster to itself since any reaction will affect it
	setReactionConnectivity(id);
	setDissociationConnectivity(id);
	setReactionConnectivity(momId[0]);
	setDissociationConnectivity(momId[0]);

	// Loop over all the reacting pairs
	for (auto it = effReactingList.begin(); it != effReactingList.end(); ++it) {
		// The cluster is connecting to both clusters in the pair
		setReactionConnectivity((*it).first->getId());
		setReactionConnectivity((*it).first->getMomentId());
		setReactionConnectivity((*it).second->getId());
		setReactionConnectivity((*it).second->getMomentId());
	}

	// Loop over all the combining pairs
	for (auto it = effCombiningList.begin(); it != effCombiningList.end();
			++it) {
		// The cluster is connecting to the combining cluster
		setReactionConnectivity((*it).first->getId());
		setReactionConnectivity((*it).first->getMomentId());
	}

	// Loop over all the dissociating pairs
	for (auto it = effDissociatingList.begin(); it != effDissociatingList.end();
			++it) {
		// The cluster is connecting to the combining cluster
		setDissociationConnectivity((*it).first->getId());
		setDissociationConnectivity((*it).first->getMomentId());
	}

	// Don't loop on the effective emission pairs because
	// this cluster is not connected to them

	// Initialize the partial vector for the moment
	int dof = network.getDOF();
	momentPartials.resize(dof, 0.0);

	return;
}

double AlloySuperCluster::getTotalFlux(int i) {
	// Initialize the moment flux
	momentFlux = 0.0;

	// Get the fluxes
	double prodFlux = getProductionFlux(i);
	double dissFlux = getDissociationFlux(i);
	double combFlux = getCombinationFlux(i);
	double emissFlux = getEmissionFlux(i);

	return prodFlux - combFlux + dissFlux - emissFlux;
}

double AlloySuperCluster::getDissociationFlux(int xi) {
	// Initial declarations
	double flux = 0.0, value = 0.0;
	AlloyCluster *dissociatingCluster = nullptr;

	// Loop over all the dissociating pairs
	for (auto it = effDissociatingList.begin(); it != effDissociatingList.end();
			++it) {
		// Get the dissociating clusters
		dissociatingCluster = (*it).first;
		double l0A = dissociatingCluster->getConcentration(0.0);
		double l1A = dissociatingCluster->getMoment();
		// Update the flux
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		flux += value * ((*it).a00 * l0A + (*it).a10 * l1A);
		// Compute the moment fluxes
		momentFlux += value * ((*it).a01 * l0A + (*it).a11 * l1A);
	}

	// Return the flux
	return flux;
}

double AlloySuperCluster::getEmissionFlux(int xi) {
	// Initial declarations
	double flux = 0.0, value = 0.0;

	// Loop over all the emission pairs
	for (auto it = effEmissionList.begin(); it != effEmissionList.end(); ++it) {
		// Update the flux
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		flux += value * ((*it).a00 * l0 + (*it).a10 * l1);
		// Compute the moment fluxes
		momentFlux -= value * ((*it).a01 * l0 + (*it).a11 * l1);
	}

	return flux;
}

double AlloySuperCluster::getProductionFlux(int xi) {
	// Local declarations
	double flux = 0.0, value = 0.0;
	AlloyCluster *firstReactant = nullptr, *secondReactant = nullptr;

	// Loop over all the reacting pairs
	for (auto it = effReactingList.begin(); it != effReactingList.end(); ++it) {
		// Get the two reacting clusters
		firstReactant = (*it).first;
		secondReactant = (*it).second;
		double l0A = firstReactant->getConcentration();
		double l0B = secondReactant->getConcentration();
		double l1A = firstReactant->getMoment();
		double l1B = secondReactant->getMoment();
		// Update the flux
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		flux += value
				* ((*it).a000 * l0A * l0B + (*it).a010 * l0A * l1B
						+ (*it).a100 * l1A * l0B + (*it).a110 * l1A * l1B);
		// Compute the moment flux
		momentFlux += value
				* ((*it).a001 * l0A * l0B + (*it).a011 * l0A * l1B
						+ (*it).a101 * l1A * l0B + (*it).a111 * l1A * l1B);
	}

	// Return the production flux
	return flux;
}

double AlloySuperCluster::getCombinationFlux(int xi) {
	// Local declarations
	double flux = 0.0, value = 0.0;
	AlloyCluster *combiningCluster = nullptr;

	// Loop over all the combining clusters
	for (auto it = effCombiningList.begin(); it != effCombiningList.end();
			++it) {
		// Get the two reacting clusters
		combiningCluster = (*it).first;
		double l0A = combiningCluster->getConcentration();
		double l1A = combiningCluster->getMoment();
		// Update the flux
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		flux += value
				* ((*it).a000 * l0A * l0 + (*it).a100 * l0A * l1
						+ (*it).a010 * l1A * l0 + (*it).a110 * l1A * l1);
		// Compute the moment flux
		momentFlux -= value
				* ((*it).a001 * l0A * l0 + (*it).a101 * l0A * l1
						+ (*it).a011 * l1A * l0 + (*it).a111 * l1A * l1);
	}

	return flux;
}

void AlloySuperCluster::getPartialDerivatives(std::vector<double> & partials,
		int i) const {
	// Reinitialize the moment partial derivatives vector
	std::fill(momentPartials.begin(), momentPartials.end(), 0.0);

	// Get the partial derivatives for each reaction type
	getProductionPartialDerivatives(partials, i);
	getCombinationPartialDerivatives(partials, i);
	getDissociationPartialDerivatives(partials, i);
	getEmissionPartialDerivatives(partials, i);

	return;
}

void AlloySuperCluster::getProductionPartialDerivatives(
		std::vector<double> & partials, int xi) const {
	// Initial declarations
	double value = 0.0;
	int index = 0;
	AlloyCluster *firstReactant = nullptr, *secondReactant = nullptr;

	// Production
	// A + B --> D, D being this cluster
	// The flux for D is
	// F(C_D) = k+_(A,B)*C_A*C_B
	// Thus, the partial derivatives
	// dF(C_D)/dC_A = k+_(A,B)*C_B
	// dF(C_D)/dC_B = k+_(A,B)*C_A

	// Loop over all the reacting pairs
	for (auto it = effReactingList.begin(); it != effReactingList.end(); ++it) {
		// Get the two reacting clusters
		firstReactant = (*it).first;
		secondReactant = (*it).second;
		double l0A = firstReactant->getConcentration();
		double l0B = secondReactant->getConcentration();
		double l1A = firstReactant->getMoment();
		double l1B = secondReactant->getMoment();

		// Compute the contribution from the first part of the reacting pair
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		index = firstReactant->getId() - 1;
		partials[index] += value * ((*it).a000 * l0B + (*it).a010 * l1B);
		momentPartials[index] += value * ((*it).a001 * l0B + (*it).a011 * l1B);
		index = firstReactant->getMomentId() - 1;
		partials[index] += value * ((*it).a100 * l0B + (*it).a110 * l1B);
		momentPartials[index] += value * ((*it).a101 * l0B + (*it).a111 * l1B);
		// Compute the contribution from the second part of the reacting pair
		index = secondReactant->getId() - 1;
		partials[index] += value * ((*it).a000 * l0A + (*it).a100 * l1A);
		momentPartials[index] += value * ((*it).a001 * l0A + (*it).a101 * l1A);
		index = secondReactant->getMomentId() - 1;
		partials[index] += value * ((*it).a010 * l0A + (*it).a110 * l1A);
		momentPartials[index] += value * ((*it).a011 * l0A + (*it).a111 * l1A);
	}

	return;
}

void AlloySuperCluster::getCombinationPartialDerivatives(
		std::vector<double> & partials, int xi) const {
	// Initial declarations
	int index = 0;
	AlloyCluster *cluster = nullptr;
	double value = 0.0;

	// Combination
	// A + B --> D, A being this cluster
	// The flux for A is outgoing
	// F(C_A) = - k+_(A,B)*C_A*C_B
	// Thus, the partial derivatives
	// dF(C_A)/dC_A = - k+_(A,B)*C_B
	// dF(C_A)/dC_B = - k+_(A,B)*C_A

	// Loop over all the combining clusters
	for (auto it = effCombiningList.begin(); it != effCombiningList.end();
			++it) {
		// Get the two reacting clusters
		cluster = (*it).first;
		double l0A = cluster->getConcentration();
		double l1A = cluster->getMoment();

		// Compute the contribution from the combining cluster
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		index = cluster->getId() - 1;
		partials[index] -= value * ((*it).a000 * l0 + (*it).a100 * l1);
		momentPartials[index] -= value * ((*it).a001 * l0 + (*it).a101 * l1);
		index = cluster->getMomentId() - 1;
		partials[index] -= value * ((*it).a010 * l0 + (*it).a110 * l1);
		momentPartials[index] -= value * ((*it).a011 * l0 + (*it).a111 * l1);
		// Compute the contribution from this cluster
		index = id - 1;
		partials[index] -= value * ((*it).a000 * l0A + (*it).a010 * l1A);
		momentPartials[index] -= value * ((*it).a001 * l0A + (*it).a011 * l1A);
		index = momId[0] - 1;
		partials[index] -= value * ((*it).a100 * l0A + (*it).a110 * l1A);
		momentPartials[index] -= value * ((*it).a101 * l0A + (*it).a111 * l1A);
	}

	return;
}

void AlloySuperCluster::getDissociationPartialDerivatives(
		std::vector<double> & partials, int xi) const {
	// Initial declarations
	int index = 0;
	AlloyCluster *cluster = nullptr;
	double value = 0.0;

	// Dissociation
	// A --> B + D, B being this cluster
	// The flux for B is
	// F(C_B) = k-_(B,D)*C_A
	// Thus, the partial derivatives
	// dF(C_B)/dC_A = k-_(B,D)

	// Loop over all the dissociating pairs
	for (auto it = effDissociatingList.begin(); it != effDissociatingList.end();
			++it) {
		// Get the dissociating clusters
		cluster = (*it).first;

		// Compute the contribution from the dissociating cluster
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		index = cluster->getId() - 1;
		partials[index] += value * ((*it).a00);
		momentPartials[index] += value * ((*it).a01);
		index = cluster->getMomentId() - 1;
		partials[index] += value * ((*it).a10);
		momentPartials[index] += value * ((*it).a11);
	}

	return;
}

void AlloySuperCluster::getEmissionPartialDerivatives(
		std::vector<double> & partials, int xi) const {
	// Initial declarations
	int index = 0;
	double value = 0.0;

	// Emission
	// A --> B + D, A being this cluster
	// The flux for A is
	// F(C_A) = - k-_(B,D)*C_A
	// Thus, the partial derivatives
	// dF(C_A)/dC_A = - k-_(B,D)

	// Loop over all the emission pairs
	for (auto it = effEmissionList.begin(); it != effEmissionList.end(); ++it) {
		// Compute the contribution from the dissociating cluster
		value = (*it).reaction.kConstant[xi] / (double) nTot;
		index = id - 1;
		partials[index] -= value * ((*it).a00);
		momentPartials[index] -= value * ((*it).a01);
		index = momId[0] - 1;
		partials[index] -= value * ((*it).a10);
		momentPartials[index] -= value * ((*it).a11);
	}

	return;
}

void AlloySuperCluster::getMomentPartialDerivatives(
		std::vector<double> & partials) const {
	// Loop on the size of the vector
	for (int i = 0; i < partials.size(); i++) {
		// Set to the values that were already computed
		partials[i] = momentPartials[i];
	}

	return;
}

std::vector<std::vector<double> > AlloySuperCluster::getProdVector() const {
	// Initial declarations
	std::vector<std::vector<double> > toReturn;

	// Loop on the reacting pairs
	std::for_each(effReactingList.begin(), effReactingList.end(),
			[&toReturn](SuperClusterProductionPair const& currPair) {
				// Build the vector containing ids and rates
				std::vector<double> tempVec;
				tempVec.push_back(currPair.first->getId() - 1);
				tempVec.push_back(currPair.second->getId() - 1);
				tempVec.push_back(currPair.a000);
				tempVec.push_back(currPair.a001);
				tempVec.push_back(currPair.a100);
				tempVec.push_back(currPair.a101);
				tempVec.push_back(currPair.a010);
				tempVec.push_back(currPair.a011);
				tempVec.push_back(currPair.a110);
				tempVec.push_back(currPair.a111);

				// Add it to the main vector
				toReturn.push_back(tempVec);
			});

	return toReturn;
}

std::vector<std::vector<double> > AlloySuperCluster::getCombVector() const {
	// Initial declarations
	std::vector<std::vector<double> > toReturn;

	// Loop on the combining reactants
	std::for_each(effCombiningList.begin(), effCombiningList.end(),
			[&toReturn](SuperClusterProductionPair const& currPair) {
				// Build the vector containing ids and rates
				std::vector<double> tempVec;
				tempVec.push_back(currPair.first->getId() - 1);
				tempVec.push_back(currPair.a000);
				tempVec.push_back(currPair.a001);
				tempVec.push_back(currPair.a100);
				tempVec.push_back(currPair.a101);
				tempVec.push_back(currPair.a010);
				tempVec.push_back(currPair.a011);
				tempVec.push_back(currPair.a110);
				tempVec.push_back(currPair.a111);

				// Add it to the main vector
				toReturn.push_back(tempVec);
			});

	return toReturn;
}

std::vector<std::vector<double> > AlloySuperCluster::getDissoVector() const {
	// Initial declarations
	std::vector<std::vector<double> > toReturn;

	// Loop on the dissociating pairs
	std::for_each(effDissociatingList.begin(), effDissociatingList.end(),
			[&toReturn](SuperClusterDissociationPair const& currPair) {
				// Build the vector containing ids and rates
				std::vector<double> tempVec;
				tempVec.push_back(currPair.first->getId() - 1);
				tempVec.push_back(currPair.second->getId() - 1);
				tempVec.push_back(currPair.a00);
				tempVec.push_back(currPair.a01);
				tempVec.push_back(currPair.a10);
				tempVec.push_back(currPair.a11);

				// Add it to the main vector
				toReturn.push_back(tempVec);
			});

	return toReturn;
}

std::vector<std::vector<double> > AlloySuperCluster::getEmitVector() const {
	// Initial declarations
	std::vector<std::vector<double> > toReturn;

	// Loop on the emitting pairs
	std::for_each(effEmissionList.begin(), effEmissionList.end(),
			[&toReturn](SuperClusterDissociationPair const& currPair) {
				// Build the vector containing ids and rates
				std::vector<double> tempVec;
				tempVec.push_back(currPair.first->getId() - 1);
				tempVec.push_back(currPair.second->getId() - 1);
				tempVec.push_back(currPair.a00);
				tempVec.push_back(currPair.a01);
				tempVec.push_back(currPair.a10);
				tempVec.push_back(currPair.a11);

				// Add it to the main vector
				toReturn.push_back(tempVec);
			});

	return toReturn;
}

