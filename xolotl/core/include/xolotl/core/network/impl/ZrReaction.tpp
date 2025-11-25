#pragma once

#include <xolotl/core/network/impl/SinkReaction.tpp>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace zr
{
template <typename TRegion>
KOKKOS_INLINE_FUNCTION
double
getRate(const TRegion& pairCl0Reg, const TRegion& pairCl1Reg, const double r0,
	const double r1, const double dc0, const double dc1, double rdCl[2][2],
	double p, int transitionSize)
{
	constexpr double pi = ::xolotl::core::pi;
	constexpr double rCore = ::xolotl::core::alphaZrCoreRadius;
	const double zs = 4.0 * pi * (r0 + r1 + rCore);

	using Species = typename TRegion::EnumIndex;
	xolotl::core::network::detail::Composition<typename TRegion::VectorType,
		Species>
		lo0 = pairCl0Reg.getOrigin();
	xolotl::core::network::detail::Composition<typename TRegion::VectorType,
		Species>
		lo1 = pairCl1Reg.getOrigin();

	// Determine if clusters are vacancy or interstitial and initialize
	// variables
	bool cl0IsV = lo0.isOnAxis(Species::V);
	bool cl1IsV = lo1.isOnAxis(Species::V);
	double n0 = 0; // size of cluster 0
	double n1 = 0; // size of cluster 1
	double Pl = 1.0; // Capture efficiency for diffusing defect
	double Pli = 1.0; // Capture efficiency for interstitial a-loop
	double Plv = 1.0; // Capture efficiency for vacancy a-loops

	// Determine parameters for cluster 0 based on cluster type and size
	if (cl0IsV){
		n0 = lo0[Species::V];
	}
	else if (lo0.isOnAxis(Species::Basal)){
		n0 = lo0[Species::Basal];
	}
	else{
		n0 = lo0[Species::I];
	}
	bool cl0IsLoop = (n0 > 9);
	
	// Determine parameters for cluster 1 based on cluster type and size
	if (cl1IsV){
		n1 = lo1[Species::V];
	}
	else if (lo1.isOnAxis(Species::Basal)){
		n1 = lo1[Species::Basal];
	}
	else{
		n1 = lo1[Species::I];
	}
	bool cl1IsLoop = (n1 > 9);
	
	// Cluster 0 is a dislocation loop
	if (cl0IsLoop) {
		// Define the dislocation capture radius, transition coefficient, and
		// then calculate the reaction rate
		double rd = rdCl[0][cl1IsV];
		double alpha = pow(1 + pow(r0 / (3 * (r1 + rd)), 2), -1);
		double rateSpherical = 4.0 * pi * (r0 + r1 + rd);
		double rateToroidal =
			(4.0 * pi * pi * r0) / log(1 + (8 * r0) / (r1 + rd));

		// Calculate the capture efficiency (assuming only prismatic loops)
		if (cl0IsV)
			Pl = 0.78 * pow(p, -2) + 0.66 * p - 0.44;
		else if (lo0.isOnAxis(Species::Basal)) {
			if (n0 < transitionSize)
				alpha = 1.0; // Completely spherical
			Pl = p;
		}
		else
			Pl = 0.70 * pow(p, -2) + 0.78 * p - 0.47;
		
		return ((1 - alpha) * rateToroidal * Pl + alpha * rateSpherical) *
			(dc0 + dc1);
	}

	// Cluster 1 is a dislocation loop:
	if (cl1IsLoop) {
		// Define the dislocation capture radius, transition coefficient, and
		// then calculate the reaction rate
		double rd = rdCl[1][cl0IsV];
		double alpha = pow(1 + pow(r1 / (3 * (r0 + rd)), 2), -1);
		double rateSpherical = 4.0 * pi * (r0 + r1 + rd);
		double rateToroidal =
			(4.0 * pi * pi * r1) / log(1 + (8 * r1) / (r0 + rd));

		// Calculate the capture efficiency (assuming only prismatic loops)
		if (cl1IsV)
			Pl = 0.78 * pow(p, -2) + 0.66 * p - 0.44;
		else if (lo1.isOnAxis(Species::Basal)) {
			if (n1 < transitionSize)
				alpha = 1.0; // Completely spherical
			Pl = p;
		}
		else
			Pl = 0.70 * pow(p, -2) + 0.78 * p - 0.47;

		return ((1 - alpha) * rateToroidal * Pl + alpha * rateSpherical) *
			(dc0 + dc1);
	}

	// None of the clusters are loops (interaction is based on spherical volume)
	return zs * (dc0 + dc1);
}
} // namespace zr

KOKKOS_INLINE_FUNCTION
double
ZrProductionReaction::getRateForProduction(IndexType gridIndex)
{
	auto cl0 = this->_clusterData->getCluster(_reactants[0]);
	auto cl1 = this->_clusterData->getCluster(_reactants[1]);

	// Create an array with all possible dislocation capture radii
	// rdCl = {(rdI for cl0, rdV for cl0), (rdI for cl1, rdV for cl1)}
	double rdCl[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
	
	if (not isLargeClusterReaction){
		
		double r0 = cl0.getReactionRadius();
		double r1 = cl1.getReactionRadius();

		double dc0 = cl0.getDiffusionCoefficient(gridIndex);
		double dc1 = cl1.getDiffusionCoefficient(gridIndex);

		
		// Determine which cluster is mobile and retrieve its anisotropy ratio
		double p = 0;
		if (dc0 > 0)
			p = this->_clusterData->extraData.anisotropyRatio(
				_reactants[0], gridIndex);
		else if (dc1 > 0)
			p = this->_clusterData->extraData.anisotropyRatio(
				_reactants[1], gridIndex);

		
		rdCl[0][0] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[0], 0);
		rdCl[0][1] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[0], 1);
		rdCl[1][0] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[1], 0);
		rdCl[1][1] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[1], 1);
		
					 

		
		return zr::getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1, rdCl,
			p, this->_clusterData->transitionSize());
	}
	// Large Cluster Case (vac only right now)
	const auto dummyRegion = Region(Composition{});
	
	double r0 = 0.0, r1 = 0.0, dc0 = 0.0, dc1 = 0.0, p=0.0;
	Region cl0Reg = dummyRegion, cl1Reg = dummyRegion;
	auto numClusters = this->_clusterData->numClusters;
	
	if (this->_reactants[0] >= numClusters){
		auto shift = (this->_reactants[0] - numClusters)/2;
		switch (shift) {
		// Vac
		case 0:
			r0 = this->_clusterData->vacAvRad();
			cl0Reg[Species::V] = {1001, 1002};
			rdCl[0][0] = 0.79;
			rdCl[0][1] = 1.59;
			break;
		// Int
		case 1:
			r0 = this->_clusterData->intAvRad();
			cl0Reg[Species::I] = {1, 2};
			rdCl[0][0] = 1.85;
			rdCl[0][1] = 1.04;
			break;
		// Basal
		case 2:
			r0 = this->_clusterData->basalAvRad();
			cl0Reg[Species::Basal] = {1, 2};
			rdCl[0][0] = 0.787;
			rdCl[0][1] = 1.072;
			break;
		}
	}
	else {
		auto cl0 = this->_clusterData->getCluster(_reactants[0]);
		r0 = cl0.getReactionRadius();
		dc0 = cl0.getDiffusionCoefficient(gridIndex);
		//anisotropy ratio
		//double p = 0;
		if (dc0 > 0)
			p = this->_clusterData->extraData.anisotropyRatio(
				_reactants[0], gridIndex);
		cl0Reg = cl0.getRegion();
		rdCl[0][0] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[0], 0);
		rdCl[0][1] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[0], 1);
	}
	
	if (this->_reactants[1] >= numClusters){
		auto shift = (this->_reactants[1] - numClusters)/2;
		switch (shift) {
		// Vac
		case 0:
			r1 = this->_clusterData->vacAvRad();
			cl1Reg[Species::V] = {1001, 1002};
			rdCl[1][0] = 0.79;
			rdCl[1][1] = 1.59;
			break;
		// Int
		case 1:
			r1 = this->_clusterData->intAvRad();
			cl1Reg[Species::I] = {1, 2};
			rdCl[1][0] = 1.85;
			rdCl[1][1] = 1.04;
			break;
		// Basal
		case 2:
			r1 = this->_clusterData->basalAvRad();
			cl1Reg[Species::Basal] = {1, 2};
			rdCl[1][0] = 0.787;
			rdCl[1][1] = 1.072;
			break;
		}
	}
	
	else {
		auto cl1 = this->_clusterData->getCluster(_reactants[1]);
		r1 = cl1.getReactionRadius();
		dc1 = cl1.getDiffusionCoefficient(gridIndex);
		//anisotropy ratio
		//double p = 0;
		if (dc1 > 0)
			p = this->_clusterData->extraData.anisotropyRatio(
				_reactants[1], gridIndex);
		cl1Reg = cl1.getRegion();
		rdCl[1][0] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[1], 0);
		rdCl[1][1] = this->_clusterData->extraData.dislocationCaptureRadius(
			_reactants[1], 1);
	}
	
	return zr::getRate(cl0Reg, cl1Reg, r0, r1, dc0, dc1, rdCl,
			p, this->_clusterData->transitionSize());
}

KOKKOS_INLINE_FUNCTION
void
ZrProductionReaction::computeCoefficients()
{
	// Check if the large cluster is involved
	if (isLargeClusterReaction) {
		constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
		for (auto i : speciesRangeNoI) {
			this->_widths(i()) = 1.0;
		}
		this->_coefs(0, 0, 0, 0) = 1.0;
	}
	else {
		// Standard case
		Superclass::computeCoefficients();
	}
}

KOKKOS_INLINE_FUNCTION
void
ZrProductionReaction::computeFlux(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex)
{
	// Standard case
	if (not isLargeClusterReaction) {
		return Superclass::computeFlux(concentrations, fluxes, gridIndex);
	}

	// The rate need to be computed each time because it depends on the current
	// large cluster size
	auto rate = getRateForProduction(gridIndex);
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
	auto numClusters = this->_clusterData->numClusters;
	auto vacId = this->_clusterData->vacId();
	auto intId = this->_clusterData->intId();
	auto basalId = this->_clusterData->basalId();
	
	// Large vacancy cluster is one of the reactants
	if ((this->_reactants[0] - numClusters)/2 == 0 or
			(this->_reactants[1] - numClusters)/2 == 0) {
				// Get standard cluster
				auto stdClusterId = (this->_reactants[0] >= numClusters) ?
					this->_reactants[1]:
					this->_reactants[0];
				auto cl = this->_clusterData->getCluster(stdClusterId);
				auto clReg = cl.getRegion();
				auto orig = clReg.getOrigin();
				Composition comp(orig);
				// Get the SSBM cluster
				auto ssbmId = (this->_reactants[0] >= numClusters) ?
					this->_reactants[0] :
					this->_reactants[1];
				
				// Vacancy Standard Cluster Case
				if (comp[Species::V] > 0){
					//Compute flux
					double f = this->_coefs(0, 0, 0, 0) * concentrations(stdClusterId)
						* concentrations(ssbmId) * rate;
					// The standard cluster always loses the flux
					Kokkos::atomic_sub(&fluxes[stdClusterId], f);
					
					// The Large V size increases
					Kokkos::atomic_add(&fluxes[ssbmId+1], f * comp[Species::V]);
				}
				
				// Interstitial Standard Cluster Case
				if (comp[Species::I] > 0){
					//Compute flux
					double f = this->_coefs(0, 0, 0, 0) * concentrations(stdClusterId)
						* concentrations(ssbmId) * rate;
					
					// Special factor to deal with the threshold size
					double gauss = 1.0;
					
					// Special case where the product is not the single size
					if (this->_products[0] < numClusters){
						// Only if large vacancy cluster has a specific size
						auto avVac = concentrations(ssbmId+1) / concentrations(ssbmId);
						if (concentrations(ssbmId) == 0.0)
								avVac = 0.0;
						
						// Get product composition
						auto pr = this->_clusterData->getCluster(this->_products[0]);
						auto prReg = pr.getRegion();
						Composition prComp(prReg.getOrigin());
						
						// Target value for the reaction to happen
						double target = prComp[Species::V] + comp[Species::I];
						
						// Gaussian function around it
						double twoSigmaTwo = 0.1;
						gauss =
							exp(-(avVac - target) * (avVac - target) / twoSigmaTwo) /
							sqrt(::xolotl::core::pi * twoSigmaTwo);
							
						// The large cluster concentration decreases
						Kokkos::atomic_sub(&fluxes[ssbmId], f * gauss);
						// The product concentration increases
						Kokkos::atomic_add(&fluxes[this->_products[0]], f * gauss);
						
						// The V size decreases even more
						Kokkos::atomic_sub(&fluxes[this->_clusterData->vacAvId()], 
							f * (prComp[Species::V]) * gauss);
					}
					// In every case
					
					// Standard Cluster always loses the flux
					Kokkos::atomic_sub(&fluxes[stdClusterId], f * gauss);
					
					// The V size decreases
					Kokkos::atomic_sub(&fluxes[ssbmId + 1], f * comp[Species::I] * gauss);
				}
			}
			
	
	
			// Large cluster is only a product
			else {
				auto cR1 = concentrations[this->_reactants[0]];
				auto cR2 = concentrations[this->_reactants[1]];
				auto cl1 = this->_clusterData->getCluster(this->_reactants[0]);
				auto cl1Reg = cl1.getRegion();
				auto orig1 = cl1Reg.getOrigin();
				Composition comp1(orig1);
				auto cl2 = this->_clusterData->getCluster(this->_reactants[1]);
				auto cl2Reg = cl2.getRegion();
				auto orig2 = cl2Reg.getOrigin();
				Composition comp2(orig2);
				
				double f = this->_coefs(0, 0, 0, 0) * cR1 * cR2 * rate;
				
				// Vacancy case
				if (orig1.isOnAxis(Species::V) or orig2.isOnAxis(Species::V)) {
					// Compute the total size
					auto totalSize = comp1[Species::V] + comp2[Species::V];
					// Both reactants decrease
					Kokkos::atomic_sub(&fluxes[this->_reactants[0]], f);
					Kokkos::atomic_sub(&fluxes[this->_reactants[1]], f);
					// The large V cluster increases, as well as average V
					Kokkos::atomic_add(&fluxes[this->_products[0]], f);
					Kokkos::atomic_add(&fluxes[this->_products[0]+1], f * totalSize);
				}
				/*
				// Interstitial case
				if ((this->_products[0] - numClusters)/2 == 1 or
					(this->_products[1] - numClusters)/2 == 1){
					// Both reactants decrease
					Kokkos::atomic_sub(&fluxes[this->_reactants[0]], f);
					Kokkos::atomic_sub(&fluxes[this->_reactants[1]], f);
					// The large I cluster increases, as well as average I
					Kokkos::atomic_add(&fluxes[intId], f);
					Kokkos::atomic_add(&fluxes[this->_clusterData->intAvId()],
						f * (comp1[Species::I] + comp2[Species::I]));
				}
				
				// Basal case
				if ((this->_products[0] - numClusters)/2 == 2 or
					(this->_products[1] - numClusters)/2 == 2){
					// Both reactants decrease
					Kokkos::atomic_sub(&fluxes[this->_reactants[0]], f);
					Kokkos::atomic_sub(&fluxes[this->_reactants[1]], f);
					// The large Basal cluster increases, as well as average Basal
					Kokkos::atomic_add(&fluxes[basalId], f);
					if (comp1[Species::V] > 0){ // large basal cluster comes from V + B
					Kokkos::atomic_add(&fluxes[this->_clusterData->basalAvId()],
						f * (comp1[Species::V] + comp2[Species::Basal]));
					}
					if (comp2[Species::V] > 0){
					Kokkos::atomic_add(&fluxes[this->_clusterData->basalAvId()],
						f * (comp1[Species::Basal] + comp2[Species::V]));
					}
				}*/
			}
}

KOKKOS_INLINE_FUNCTION
void
ZrProductionReaction::computePartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	IndexType gridIndex)
{
	// Standard case
	if (not isLargeClusterReaction) {
		return Superclass::computePartialDerivatives(
			concentrations, values, gridIndex);
	}

	// The rate need to be computed each time because it depends on the current
	// large cluster size
	auto rate = getRateForProduction(gridIndex);

	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
	auto numClusters = this->_clusterData->numClusters;
	auto vacId = this->_clusterData->vacId();
	auto intId = this->_clusterData->intId();
	auto basalId = this->_clusterData->basalId();
	
	// Large vacancy cluster is one of the reactants
	if (this->_reactants[0] >= numClusters or
			this->_reactants[1] >= numClusters) {
				// Get standard cluster
				auto stdClusterId = (this->_reactants[0] >= numClusters) ?
					this->_reactants[1]:
					this->_reactants[0];
				auto cl = this->_clusterData->getCluster(stdClusterId);
				auto clReg = cl.getRegion();
				auto orig = clReg.getOrigin();
				Composition comp(orig);
				// Get the SSBM cluster
				auto ssbmId = (this->_reactants[0] >= numClusters) ?
					this->_reactants[0] :
					this->_reactants[1];
				// Get the concentrations
				auto stdC = concentrations(stdClusterId);
				auto vC = concentrations(ssbmId); //conc of large cluster
				
				// Vacancy Standard Cluster Case
				if (comp[Species::V] > 0){
					//Compute flux
					double f = this->_coefs(0, 0, 0, 0) * rate;
					
					// The standard cluster always loses the flux
					if (this->_reactants[0] >= numClusters){
						Kokkos::atomic_sub(&values(this->_connEntries[1][0][0][0]), f * stdC);
						Kokkos::atomic_sub(&values(this->_connEntries[1][0][1][0]), f * vC);
					}
					else {
						Kokkos::atomic_sub(&values(this->_connEntries[0][0][1][0]), f * stdC);
						Kokkos::atomic_sub(&values(this->_connEntries[0][0][0][0]), f * vC);
					}
					
					// The Large V size increases
					f = this->_coefs(0, 0, 0, 0) * rate * comp[Species::V];
					if (this->_reactants[0] >= numClusters) {
						Kokkos::atomic_add(
							&values(this->_connEntries[0][1][0][0]), f * stdC);
						Kokkos::atomic_add(
							&values(this->_connEntries[0][1][1][0]), f * vC);
					}
					else {
						Kokkos::atomic_add(
							&values(this->_connEntries[1][1][1][0]), f * stdC);
						Kokkos::atomic_add(
							&values(this->_connEntries[1][1][0][0]), f * vC);
					}
				}
			
				
				// Interstitial Standard Cluster Case
				if (comp[Species::I] > 0){
					//Compute flux
					double f = this->_coefs(0, 0, 0, 0) * rate;
					
					// Special factor to deal with the threshold size
					double gauss = 1.0;
					
					// Special case where the product is not the single size
					if (this->_products[0] < numClusters){
						// Only if large vacancy cluster has a specific size
						auto avVac = concentrations(ssbmId+1) / concentrations(ssbmId);
						if (concentrations(ssbmId) == 0.0)
								avVac = 0.0;
						
						// Get product composition
						auto pr = this->_clusterData->getCluster(this->_products[0]);
						auto prReg = pr.getRegion();
						Composition prComp(prReg.getOrigin());
						
						// Target value for the reaction to happen
						double target = prComp[Species::V] + comp[Species::I];
						
						// Gaussian function around it
						double twoSigmaTwo = 0.1;
						double gauss = exp(-(avVac - target) * (avVac - target) / twoSigmaTwo)/
							sqrt(::xolotl::core::pi * twoSigmaTwo);
						
						//Update rate
						f = this->_coefs(0, 0, 0, 0) * rate * gauss;
						
						// The large cluster concentration decreases
						if (this->_reactants[0] >= numClusters) {
							Kokkos::atomic_sub(
								&values(this->_connEntries[0][0][0][0]), f * stdC);
							Kokkos::atomic_sub(
								&values(this->_connEntries[0][0][1][0]), f * vC);
						}
						else {
							Kokkos::atomic_sub(
								&values(this->_connEntries[1][0][1][0]), f * stdC);
							Kokkos::atomic_sub(
								&values(this->_connEntries[1][0][0][0]), f * vC);
						}
						
						// The product concentration increases
						if (this->_reactants[0] >= numClusters) {
							Kokkos::atomic_add(
								&values(this->_connEntries[2][0][0][0]), f * stdC);
							Kokkos::atomic_add(
								&values(this->_connEntries[2][0][1][0]), f * vC);
					}
					else {
						Kokkos::atomic_add(
							&values(this->_connEntries[2][0][1][0]), f * stdC);
						Kokkos::atomic_add(
							&values(this->_connEntries[2][0][0][0]), f * vC);
					}
					// The V size decreases even more
					f = this->_coefs(0, 0, 0, 0) * rate *
						prComp[Species::V] * gauss;
					if (this->_reactants[0] >= numClusters) {
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][1][0][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][1][1][0]), f * vC);
					}
					else {
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][1][1][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][1][0][0]), f * vC);
					}
				}
				// In every case
				
				// The standard cluster always loses the flux
				f = this->_coefs(0, 0, 0, 0) * rate * gauss;
					if (this->_reactants[0] >= numClusters){
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][0][0][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][0][1][0]), f * vC);
					}
					else {
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][0][1][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][0][0][0]), f * vC);
					}
					
					// The large V size decreases
					f = this->_coefs(0, 0, 0, 0) * rate * comp[Species::I] * gauss;
					if (this->_reactants[0] >= numClusters) {
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][1][0][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][1][1][0]), f * vC);
					}
					else {
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][1][1][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][1][0][0]), f * vC);
					}
				}
			}
		// Large bubble is one of the product
		else {
			auto cR1 = concentrations[this->_reactants[0]];
			auto cR2 = concentrations[this->_reactants[1]];
			auto cl1 = this->_clusterData->getCluster(this->_reactants[0]);
			auto cl1Reg = cl1.getRegion();
			auto orig1 = cl1Reg.getOrigin();
			Composition comp1(orig1);
			auto cl2 = this->_clusterData->getCluster(this->_reactants[1]);
			auto cl2Reg = cl2.getRegion();
			auto orig2 = cl2Reg.getOrigin();
			Composition comp2(orig2);

			double f = this->_coefs(0, 0, 0, 0) * rate;

			// Vacancy case
			if (orig1.isOnAxis(Species::V) or orig2.isOnAxis(Species::V)) {
				// Both reactants decrease
				Kokkos::atomic_sub(
					&values(this->_connEntries[0][0][0][0]), f * cR2);
				Kokkos::atomic_sub(
					&values(this->_connEntries[1][0][0][0]), f * cR2);
				Kokkos::atomic_sub(
					&values(this->_connEntries[0][0][1][0]), f * cR1);
				Kokkos::atomic_sub(
					&values(this->_connEntries[1][0][1][0]), f * cR1);

				// The large cluster increases, as well as average V
				Kokkos::atomic_add(
					&values(this->_connEntries[2][0][0][0]), f * cR2);
				Kokkos::atomic_add(
					&values(this->_connEntries[2][0][1][0]), f * cR1);
				f = this->_coefs(0, 0, 0, 0) * rate *	(comp1[Species::V] + comp2[Species::V]);
			
				Kokkos::atomic_add(
					&values(this->_connEntries[2][1][0][0]), f * cR2);
				Kokkos::atomic_add(
					&values(this->_connEntries[2][1][1][0]), f * cR1);
			}
		}
	}


KOKKOS_INLINE_FUNCTION
double
ZrDissociationReaction::getRateForProduction(IndexType gridIndex)
{
	auto cl0 = this->_clusterData->getCluster(_products[0]);
	auto cl1 = this->_clusterData->getCluster(_products[1]);

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	double dc0 = cl0.getDiffusionCoefficient(gridIndex);
	double dc1 = cl1.getDiffusionCoefficient(gridIndex);


	// Determine which cluster is mobile and retrieve its anisotropy ratio
	double p = 0;
	if (dc0 > 0)
		p = this->_clusterData->extraData.anisotropyRatio(
			_products[0], gridIndex);
	else if (dc1 > 0)
		p = this->_clusterData->extraData.anisotropyRatio(
			_products[1], gridIndex);

	// Create an array with all possible dislocation capture radii
	// rdCl = {(rdI for cl0, rdV for cl0), (rdI for cl1, rdV for cl1)}
	double rdCl[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
	rdCl[0][0] =
		this->_clusterData->extraData.dislocationCaptureRadius(_products[0], 0);
	rdCl[0][1] =
		this->_clusterData->extraData.dislocationCaptureRadius(_products[0], 1);
	rdCl[1][0] =
		this->_clusterData->extraData.dislocationCaptureRadius(_products[1], 0);
	rdCl[1][1] =
		this->_clusterData->extraData.dislocationCaptureRadius(_products[1], 1);

	return zr::getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1, rdCl,
		p, this->_clusterData->transitionSize());
}

KOKKOS_INLINE_FUNCTION
double
ZrDissociationReaction::computeBindingEnergy(double time)
{
	using Species = typename Superclass::Species;
	using Composition = typename Superclass::Composition;

	double be = 5.0;

	auto cl = this->_clusterData->getCluster(this->_reactant);
	auto prod1 = this->_clusterData->getCluster(this->_products[0]);
	auto prod2 = this->_clusterData->getCluster(this->_products[1]);

	auto clReg = cl.getRegion();
	auto prod1Reg = prod1.getRegion();
	auto prod2Reg = prod2.getRegion();
	Composition lo = clReg.getOrigin();
	Composition hi = clReg.getUpperLimitPoint();
	Composition prod1Comp = prod1Reg.getOrigin();
	Composition prod2Comp = prod2Reg.getOrigin();
	double Efn1 = 0.0;
	double Efn2 = 0.0;

	if (lo.isOnAxis(Species::V)) {
		double n = (double)(lo[Species::V] + hi[Species::V] - 1) / 2.0;
		if (prod1Comp.isOnAxis(Species::V) || prod2Comp.isOnAxis(Species::V)) {
			// For small sizes, use MD power-law fits
			// For large sizes, use Varvenne-provided formation energies
			if (n < 18)
				be = 2.03 - 1.9 * (pow(n, 0.84) - pow(n - 1.0, 0.84));
			else if (n < 66)
				be = 2.03 - 3.4 * (pow(n, 0.70) - pow(n - 1.0, 0.70));
			else if (n < 925) {
				Efn1 = 0.11 * n + 1.741 * (sqrt(n)) * log(4.588 * sqrt(n));
				Efn2 = 0.11 * (n - 1) +
					1.741 * (sqrt(n - 1)) * log(4.588 * sqrt(n - 1));
				be = 2.03 - (Efn1 - Efn2);
			}
			else {
				Efn1 = 2 * 3.14 * 1.1 * 1.69 * 0.25 * sqrt(n) *
					log(1.69 * sqrt(n) / 0.23);
				Efn2 = 2 * 3.14 * 1.1 * 1.69 * 0.25 * sqrt(n - 1) *
					log(1.69 * sqrt(n - 1) / 0.23);
				be = 2.03 - (Efn1 - Efn2);
			}
		}
	}

	// adding basal
	else if (lo.isOnAxis(Species::Basal)) {
		// Time dependence
		double x = time;
		double cp = x;
		double gamma = cp;

		double n = (double)(lo[Species::Basal] + hi[Species::Basal] - 1) / 2.0;
		/*if (prod1Comp.isOnAxis(Species::Basal) ||
			prod2Comp.isOnAxis(Species::Basal)) */
		{
			if (n < this->_clusterData->transitionSize()) {
				be = 1.762 +
					((5.352 * sqrt(n - 1) + 0.122 * (n - 1) + 0.154 * (n - 1) -
						 5.3) -
						(5.352 * sqrt(n) + 0.122 * n + 0.154 * n -
							5.3)); // With basal SFE
				// be = 1.762 + ((5.352*sqrt(n-1)+0.122*(n-1)-5.3) -
				// (5.352*sqrt(n)+0.122*n-5.3)); //Without basal SFE
			}
			else if (n < 200)
				be = 2.03 +
					2.87 *
						(sqrt(n - 1) * log(1.50 * sqrt(n - 1)) -
							sqrt(n) * log(1.50 * sqrt(n))) -
					9.08 * 0.0171; // With SFE
			// be = 2.03 + 2.87 * (sqrt(n-1)*log(1.50*sqrt(n-1)) -
			// sqrt(n)*log(1.50*sqrt(n))); //Without basal SFE
			else
				be = 2.03 +
					3.02 *
						(sqrt(n - 1) * log(1.64 * sqrt(n - 1)) -
							sqrt(n) * log(1.64 * sqrt(n))) -
					9.08 * 0.00918; // With SFE
			// be = 2.03 + 3.02 * (sqrt(n-1)*log(1.64*sqrt(n-1)) -
			// sqrt(n)*log(1.64*sqrt(n))) ; //Without basal SFE
		}
	}

	else if (lo.isOnAxis(Species::I)) {
		double n = (double)(lo[Species::I] + hi[Species::I] - 1) / 2.0;
		if (prod1Comp.isOnAxis(Species::I) || prod2Comp.isOnAxis(Species::I)) {
			if (n < 7)
				be = 2.94 - 2.8 * (pow(n, 0.81) - pow(n - 1.0, 0.81));
			else
				be = 2.94 - 4.6 * (pow(n, 0.66) - pow(n - 1.0, 0.66));
		}
	}

	return util::max(0.1, be);
}

KOKKOS_INLINE_FUNCTION
double
ZrSinkReaction::computeRate(IndexType gridIndex, double time)
{
	using Species = typename Superclass::Species;
	using Composition = typename Superclass::Composition;

	auto cl = this->_clusterData->getCluster(_reactant);
	double dc = cl.getDiffusionCoefficient(gridIndex);
	double anisotropy =
		this->_clusterData->extraData.anisotropyRatio(_reactant, gridIndex);
	double dislocationDensity = this->_clusterData->dislocationDensity();
	double alphaZrASinkStrength = dislocationDensity *
		0.7631579; // fraction of total dislocation density for type A
				   // dislocations from single crystal data (7.25/9.5)
	double alphaZrCSinkStrength = dislocationDensity * 0.2368421;

	auto clReg = cl.getRegion();
	Composition lo = clReg.getOrigin();

	if (lo.isOnAxis(Species::V)) {
		return dc * 1.0 *
			(alphaZrCSinkStrength * anisotropy +
				alphaZrASinkStrength / (anisotropy * anisotropy));
	}

	// 1-D diffusers are assumed to only interact with <a>-type edge dislocation
	// lines The anisotropy factor is assumed equal to 1.0 in this case
	else if (lo.isOnAxis(Species::I)) {
		if (lo[Species::I] < 9) {
			return dc * 1.1 *
				(alphaZrCSinkStrength * anisotropy +
					alphaZrASinkStrength / (anisotropy * anisotropy));
		}
		else if (lo[Species::I] == 9) {
			return dc * 1.1 * (alphaZrASinkStrength);
		}
	}

	return 1.0;
}
} // namespace network
} // namespace core
} // namespace xolotl
