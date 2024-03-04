#pragma once

#include <boost/math/special_functions/gamma.hpp>

#include <xolotl/core/network/PSIClusterGenerator.h>
#include <xolotl/core/network/impl/BurstingReaction.tpp>
#include <xolotl/core/network/impl/Reaction.tpp>
#include <xolotl/core/network/impl/SinkReaction.tpp>
#include <xolotl/core/network/impl/TrapMutationReaction.tpp>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace psi
{
template <typename TRegion>
KOKKOS_INLINE_FUNCTION
double
getRate(const TRegion& pairCl0Reg, const TRegion& pairCl1Reg, const double r0,
	const double r1, const double dc0, const double dc1,
	const double temperature, const double atomicVolume)
{
	constexpr double pi = ::xolotl::core::pi;

	double kPlus = 4.0 * pi * (r0 + r1) * (dc0 + dc1);

	using Species = typename TRegion::EnumIndex;
	xolotl::core::network::detail::Composition<typename TRegion::VectorType,
		Species>
		lo0 = pairCl0Reg.getOrigin();
	xolotl::core::network::detail::Composition<typename TRegion::VectorType,
		Species>
		lo1 = pairCl1Reg.getOrigin();

	// If both are pure helium
	if (lo0.isOnAxis(Species::He) and lo1.isOnAxis(Species::He))
		return kPlus;

	// If no helium is involved
	if (lo0[Species::He] == 0 or lo1[Species::He] == 0)
		return kPlus;

	// He + HeV
	xolotl::core::network::detail::Composition<typename TRegion::VectorType,
		Species>
		hi0 = pairCl0Reg.getUpperLimitPoint();
	xolotl::core::network::detail::Composition<typename TRegion::VectorType,
		Species>
		hi1 = pairCl1Reg.getUpperLimitPoint();

	// Get the pure helium and bubble radii
	double rHe = (lo0[Species::V] == 0) ? r0 : r1;
	double rB = (lo0[Species::V] == 0) ? r1 : r0;

	// Other constants
	double beta =
		(temperature > 0.0) ? 1.0 / (temperature * 1.380649e-23) : 0.0;
	double T = temperature - 273.15;
	double nu = 0.28005 + 0.05744e-4 * T + 0.54e-8 * T * T; // Poisson ratio
	double gamma = 8.27 - 0.0032 * temperature; // W/He interface free energy

	// Relaxation radius
	auto nHe = (lo0[Species::V] == 0) ? lo0[Species::He] : lo1[Species::He];
	double factor = 0.0;
	switch (nHe) {
	case 1:
		factor = 0.36;
		break;
	case 2:
		factor = 0.8;
		break;
	case 3:
		factor = 1.16;
		break;
	case 4:
		factor = 1.65;
		break;
	case 5:
		factor = 2.03;
		break;
	case 6:
		factor = 2.5; // TODO: update
		break;
	case 7:
		factor = 3.0; // TODO: update
		break;
	}
	double relaxVolume = factor * atomicVolume * 1.0e-27; // m^3

	// Pressure
	double vB =
		(4.0 / 3.0) * ::xolotl::core::pi * rB * rB * rB * 1.0e-27; // m^3
	double nHeB = (lo0[Species::V] == 0) ?
		(double)(lo1[Species::He] + hi1[Species::He] - 1) / 2.0 :
		(double)(lo0[Species::He] + hi0[Species::He] - 1) / 2.0;
	double coeff_c =
		(22.575 + 0.0064655 * temperature - 7.2645 * sqrt(1.0 / temperature));
	double coeff_b = (-12.483 - 0.024549 * temperature);
	double coeff_a = (1.0596 + 0.10604 * temperature -
		19.641 * sqrt(1.0 / temperature) + (189.84 / temperature));
	double coeff_d = -(vB / nHeB) * 6.023 * 1.0e29;

	double d = coeff_d / coeff_a;
	double c = coeff_c / coeff_a;
	double b = coeff_b / coeff_a;
	double a = b;
	b = c, c = d;

	double Q = (a * a - 3.0 * b) / 9.0;
	double R = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;

	double pMLB = 0.0;
	if (R * R < Q * Q * Q) {
		double theta = acos(R / sqrt(Q * Q * Q));
		pMLB = -2.0 * sqrt(Q) * cos(theta / 3.0) - (a / 3.0);
	}
	else {
		double A = -((R > 0) - (R < 0)) *
			pow(fabs(R) + sqrt(R * R - Q * Q * Q), 1.0 / 3.0);
		double B = Q / A;
		pMLB = (A + B) - (a / 3.0);
	}

	double P = 1.0e8 * (1.0 / (pMLB * pMLB * pMLB)); // Pressure P is in Pa

	// A(T)
	double rBCube = rB * rB * rB * 1.0e-27; // m^3
	double AT = (relaxVolume / 2.0) * ((1.0 + nu) / (1.0 - 2.0 * nu)) *
		(P - (2.0 * gamma) / (rB * 1.0e-9)) * rBCube;

	// Skip smaller clusters
	double nV = (lo0[Species::V] + hi0[Species::V] + lo1[Species::V] +
					hi1[Species::V] - 2) /
		2.0;
	if ((P - (2.0 * gamma) / (rB * 1.0e-9)) <= 0.0) {
		//		std::cout << nHe << " " << lo0[Species::He] + lo1[Species::He] -
		// nHe << " " << hi0[Species::He] + hi1[Species::He] - nHe - 2
		//				 << " " << lo0[Species::V] + lo1[Species::V] << " " <<
		// hi0[Species::V] + hi1[Species::V] - 2 << " " << rB << " " << 1 <<
		// std::endl;
		return kPlus;
	}

	// Incomplete gamma
	auto upper = boost::math::tgamma(1.0 / 3.0, beta * AT / rBCube);
	auto full = std::tgamma(1.0 / 3.0);

	// Effective radius
	double rEff = 3.0e9 * pow(beta * AT, 1.0 / 3.0) / (full - upper); // nm

	// Remove changes that are too large
	if (rEff / rB > 6.0)
		return kPlus;

	//	std::cout << nHe << " " << lo0[Species::He] + lo1[Species::He] - nHe <<
	//" " << hi0[Species::He] + hi1[Species::He] - nHe - 2
	//			 << " " << lo0[Species::V] + lo1[Species::V] << " " <<
	// hi0[Species::V] + hi1[Species::V] - 2 << " " << rEff << " " << rEff / rB
	//<< std::endl;

	return 4.0 * pi * (rEff + rHe) * (dc0 + dc1);
}
} // namespace psi

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
void
PSIProductionReaction<TSpeciesEnum>::computeCoefficients()
{
	// Check if the large bubble is involved
	if (isLargeBubbleReaction) {
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

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
void
PSIProductionReaction<TSpeciesEnum>::computeFlux(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex)
{
	// Standard case
	if (not isLargeBubbleReaction) {
		return Superclass::computeFlux(concentrations, fluxes, gridIndex);
	}

	// The rate need to be computed each time because it depends on the current
	// large bubble size
	auto rate = getRateForProduction(gridIndex);

	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
	auto largeBubbleId = this->_clusterData->bubbleId();

	// Large bubble is one of the reactants
	if (this->_reactants[0] == largeBubbleId or
		this->_reactants[1] == largeBubbleId) {
		// Get the standard cluster
		auto stdClusterId = (this->_reactants[0] == largeBubbleId) ?
			this->_reactants[1] :
			this->_reactants[0];
		auto cl = this->_clusterData->getCluster(stdClusterId);
		auto clReg = cl.getRegion();
		auto orig = clReg.getOrigin();
		Composition comp(orig);

		// Compute the flux
		double f = this->_coefs(0, 0, 0, 0) * concentrations(stdClusterId) *
			concentrations(largeBubbleId) * rate;
		// The standard cluster always loses the flux
		Kokkos::atomic_sub(&fluxes[stdClusterId], f);

		// He case
		if (orig.isOnAxis(Species::He)) {
			// The average He increases
			Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvHeId()],
				f * comp[Species::He]);

			// Trap mutation case
			auto avHe = this->_clusterData->bubbleAvHe();
			auto avV = this->_clusterData->bubbleAvV();

			// Sigmoid
			double x = ((comp[Species::He] + avHe) /
						   util::getMaxHePerVLoop(avV,
							   this->_clusterData->latticeParameter(),
							   cl.getTemperature(gridIndex))) -
				1.0;
			double sigmo = 1.0 / (1.0 + std::exp(-50.0 * x));
			double nI = 1.0;

			// The other product increases
			if (this->_products[1] != Superclass::invalidIndex) {
				Kokkos::atomic_add(&fluxes[this->_products[1]], nI * f * sigmo);

				// The average V increases
				Composition prodComp(
					this->_clusterData->getCluster(this->_products[1])
						.getRegion()
						.getOrigin());
				Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvVId()],
					nI * f * prodComp[Species::I] * sigmo);
			}
			else {
				Kokkos::atomic_add(
					&fluxes[this->_clusterData->bubbleAvVId()], nI * f * sigmo);
			}
		}
		// V case
		else if (orig.isOnAxis(Species::V)) {
			// The average V increases
			Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvVId()],
				f * comp[Species::V]);
		}
		// I case
		else if (orig.isOnAxis(Species::I)) {
			auto avV = this->_clusterData->bubbleAvV();
			// Either the product is also the large bubble
			if (this->_products[0] == largeBubbleId) {
				// The average V decreases
				Kokkos::atomic_sub(&fluxes[this->_clusterData->bubbleAvVId()],
					f * comp[Species::I]);
			}
			// Or not
			else {
				// Only the case if the bubble is close to the V limit
				Composition prodComp(
					this->_clusterData->getCluster(this->_products[0])
						.getRegion()
						.getOrigin());
				if ((int)avV == comp[Species::I] + prodComp[Species::V]) {
					// The large bubble concentration decreases
					Kokkos::atomic_sub(&fluxes[largeBubbleId], f);
					// The product increases
					Kokkos::atomic_add(&fluxes[this->_products[0]], f);
					// The average He and V decrease
					Kokkos::atomic_sub(
						&fluxes[this->_clusterData->bubbleAvHeId()],
						f * prodComp[Species::He]);
					// TODO: check the factor is correct
					Kokkos::atomic_sub(
						&fluxes[this->_clusterData->bubbleAvVId()],
						f * (prodComp[Species::V] + comp[Species::I]));
				}
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

		double f = this->_coefs(0, 0, 0, 0) * cR1 * cR2 * rate;

		// He case
		if (orig1.isOnAxis(Species::He) or orig2.isOnAxis(Species::He)) {
			// Both reactants decrease
			Kokkos::atomic_sub(&fluxes[this->_reactants[0]], f);
			Kokkos::atomic_sub(&fluxes[this->_reactants[1]], f);
			// The large bubble increases, as well as average He and V
			Kokkos::atomic_add(&fluxes[largeBubbleId], f);
			Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvHeId()],
				f * (comp1[Species::He] + comp2[Species::He]));

			// Trap mutation case
			if (this->_products[1] != Superclass::invalidIndex) {
				// The second product increases
				Kokkos::atomic_add(&fluxes[this->_products[1]], f);

				Composition prodComp(
					this->_clusterData->getCluster(this->_products[1])
						.getRegion()
						.getOrigin());
				Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvVId()],
					f *
						(comp1[Species::V] + comp2[Species::V] +
							prodComp[Species::I]));
			}
			else {
				Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvVId()],
					f * (comp1[Species::V] + comp2[Species::V]));
			}

			return;
		}
		// V case
		if (orig1.isOnAxis(Species::V) or orig2.isOnAxis(Species::V)) {
			// Both reactants decrease
			Kokkos::atomic_sub(&fluxes[this->_reactants[0]], f);
			Kokkos::atomic_sub(&fluxes[this->_reactants[1]], f);
			// The large bubble increases, as well as average He and V
			Kokkos::atomic_add(&fluxes[largeBubbleId], f);
			Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvHeId()],
				f * (comp1[Species::He] + comp2[Species::He]));
			Kokkos::atomic_add(&fluxes[this->_clusterData->bubbleAvVId()],
				f * (comp1[Species::V] + comp2[Species::V]));

			return;
		}
	}
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
void
PSIProductionReaction<TSpeciesEnum>::computePartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	IndexType gridIndex)
{
	// Standard case
	if (not isLargeBubbleReaction) {
		return Superclass::computePartialDerivatives(
			concentrations, values, gridIndex);
	}

	// The rate need to be computed each time because it depends on the current
	// large bubble size
	auto rate = getRateForProduction(gridIndex);

	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
	auto largeBubbleId = this->_clusterData->bubbleId();

	// Large bubble is one of the reactants
	if (this->_reactants[0] == largeBubbleId or
		this->_reactants[1] == largeBubbleId) {
		// Get the standard cluster
		auto stdClusterId = (this->_reactants[0] == largeBubbleId) ?
			this->_reactants[1] :
			this->_reactants[0];
		auto cl = this->_clusterData->getCluster(stdClusterId);
		auto clReg = cl.getRegion();
		auto orig = clReg.getOrigin();
		Composition comp(orig);
		auto stdC = concentrations(stdClusterId);
		auto bC = concentrations(largeBubbleId);

		// Compute the flux
		double f = this->_coefs(0, 0, 0, 0) * rate;
		// The standard cluster always loses the flux
		if (this->_reactants[0] == largeBubbleId) {
			Kokkos::atomic_sub(
				&values(this->_connEntries[1][0][0][0]), f * stdC);
			Kokkos::atomic_sub(&values(this->_connEntries[1][0][1][0]), f * bC);
		}
		else {
			Kokkos::atomic_sub(
				&values(this->_connEntries[0][0][1][0]), f * stdC);
			Kokkos::atomic_sub(&values(this->_connEntries[0][0][0][0]), f * bC);
		}

		// He case
		if (orig.isOnAxis(Species::He)) {
			f = this->_coefs(0, 0, 0, 0) * rate * comp[Species::He];
			// The average He increases
			if (this->_reactants[0] == largeBubbleId) {
				Kokkos::atomic_add(
					&values(this->_connEntries[0][1][0][0]), f * stdC);
				Kokkos::atomic_add(
					&values(this->_connEntries[0][1][1][0]), f * bC);
			}
			else {
				Kokkos::atomic_add(
					&values(this->_connEntries[1][1][1][0]), f * stdC);
				Kokkos::atomic_add(
					&values(this->_connEntries[1][1][0][0]), f * bC);
			}

			// Trap mutation case
			auto avHe = this->_clusterData->bubbleAvHe();
			auto avV = this->_clusterData->bubbleAvV();

			// Sigmoid
			double x = ((comp[Species::He] + avHe) /
						   util::getMaxHePerVLoop(avV,
							   this->_clusterData->latticeParameter(),
							   cl.getTemperature(gridIndex))) -
				1.0;
			double sigmo = 1.0 / (1.0 + std::exp(-50.0 * x));
			double nI = 1.0;

			// The other product increases
			f = this->_coefs(0, 0, 0, 0) * rate * nI;
			if (this->_products[1] != Superclass::invalidIndex) {
				if (this->_reactants[0] == largeBubbleId) {
					Kokkos::atomic_add(&values(this->_connEntries[3][0][0][0]),
						f * sigmo * stdC);
					Kokkos::atomic_add(&values(this->_connEntries[3][0][1][0]),
						f * sigmo * bC);
				}
				else {
					Kokkos::atomic_add(&values(this->_connEntries[3][0][1][0]),
						f * sigmo * stdC);
					Kokkos::atomic_add(&values(this->_connEntries[3][0][0][0]),
						f * sigmo * bC);
				}
				// The average V increases
				Composition prodComp(
					this->_clusterData->getCluster(this->_products[1])
						.getRegion()
						.getOrigin());
				f = this->_coefs(0, 0, 0, 0) * rate * nI * prodComp[Species::I];
				if (this->_reactants[0] == largeBubbleId) {
					Kokkos::atomic_add(&values(this->_connEntries[0][2][0][0]),
						f * sigmo * stdC);
					Kokkos::atomic_add(&values(this->_connEntries[0][2][1][0]),
						f * sigmo * bC);
				}
				else {
					Kokkos::atomic_add(&values(this->_connEntries[1][2][1][0]),
						f * sigmo * stdC);
					Kokkos::atomic_add(&values(this->_connEntries[1][2][0][0]),
						f * sigmo * bC);
				}
			}
			else {
				f = this->_coefs(0, 0, 0, 0) * rate * nI;
				if (this->_reactants[0] == largeBubbleId) {
					Kokkos::atomic_add(&values(this->_connEntries[0][2][0][0]),
						f * sigmo * stdC);
					Kokkos::atomic_add(&values(this->_connEntries[0][2][1][0]),
						f * sigmo * bC);
				}
				else {
					Kokkos::atomic_add(&values(this->_connEntries[1][2][1][0]),
						f * sigmo * stdC);
					Kokkos::atomic_add(&values(this->_connEntries[1][2][0][0]),
						f * sigmo * bC);
				}
			}
		}
		// V case
		else if (orig.isOnAxis(Species::V)) {
			// The average V increases
			f = this->_coefs(0, 0, 0, 0) * rate * comp[Species::V];
			if (this->_reactants[0] == largeBubbleId) {
				Kokkos::atomic_add(
					&values(this->_connEntries[0][2][0][0]), f * stdC);
				Kokkos::atomic_add(
					&values(this->_connEntries[0][2][1][0]), f * bC);
			}
			else {
				Kokkos::atomic_add(
					&values(this->_connEntries[1][2][1][0]), f * stdC);
				Kokkos::atomic_add(
					&values(this->_connEntries[1][2][0][0]), f * bC);
			}
		}
		// I case
		else if (orig.isOnAxis(Species::I)) {
			auto avV = this->_clusterData->bubbleAvV();
			// Either the product is also the large bubble
			if (this->_products[0] == largeBubbleId) {
				// The average V decreases
				f = this->_coefs(0, 0, 0, 0) * rate * comp[Species::I];
				if (this->_reactants[0] == largeBubbleId) {
					Kokkos::atomic_sub(
						&values(this->_connEntries[0][2][0][0]), f * stdC);
					Kokkos::atomic_sub(
						&values(this->_connEntries[0][2][1][0]), f * bC);
				}
				else {
					Kokkos::atomic_sub(
						&values(this->_connEntries[1][2][1][0]), f * stdC);
					Kokkos::atomic_sub(
						&values(this->_connEntries[1][2][0][0]), f * bC);
				}
			}
			// Or not
			else {
				// Only the case if the bubble is close to the V limit
				Composition prodComp(
					this->_clusterData->getCluster(this->_products[0])
						.getRegion()
						.getOrigin());
				if ((int)avV == comp[Species::I] + prodComp[Species::V]) {
					// The large bubble concentration decreases
					f = this->_coefs(0, 0, 0, 0) * rate;
					if (this->_reactants[0] == largeBubbleId) {
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][0][0][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][0][1][0]), f * bC);
					}
					else {
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][0][1][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][0][0][0]), f * bC);
					}
					// The product increases
					if (this->_reactants[0] == largeBubbleId) {
						Kokkos::atomic_add(
							&values(this->_connEntries[2][0][0][0]), f * stdC);
						Kokkos::atomic_add(
							&values(this->_connEntries[2][0][1][0]), f * bC);
					}
					else {
						Kokkos::atomic_add(
							&values(this->_connEntries[2][0][1][0]), f * stdC);
						Kokkos::atomic_add(
							&values(this->_connEntries[2][0][0][0]), f * bC);
					}
					// The average He and V decrease
					f = this->_coefs(0, 0, 0, 0) * rate * prodComp[Species::He];
					if (this->_reactants[0] == largeBubbleId) {
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][1][0][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][1][1][0]), f * bC);
					}
					else {
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][1][1][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][1][0][0]), f * bC);
					}
					// TODO: check the factor is correct
					f = this->_coefs(0, 0, 0, 0) * rate *
						(prodComp[Species::V] + comp[Species::I]);
					if (this->_reactants[0] == largeBubbleId) {
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][2][0][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[0][2][1][0]), f * bC);
					}
					else {
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][2][1][0]), f * stdC);
						Kokkos::atomic_sub(
							&values(this->_connEntries[1][2][0][0]), f * bC);
					}
				}
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

		// He case
		if (orig1.isOnAxis(Species::He) or orig2.isOnAxis(Species::He)) {
			// Both reactants decrease
			Kokkos::atomic_sub(
				&values(this->_connEntries[0][0][0][0]), f * cR2);
			Kokkos::atomic_sub(
				&values(this->_connEntries[1][0][0][0]), f * cR2);
			Kokkos::atomic_sub(
				&values(this->_connEntries[0][0][1][0]), f * cR1);
			Kokkos::atomic_sub(
				&values(this->_connEntries[1][0][1][0]), f * cR1);
			// The large bubble increases, as well as average He and V
			Kokkos::atomic_add(
				&values(this->_connEntries[2][0][0][0]), f * cR2);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][0][1][0]), f * cR1);
			f = this->_coefs(0, 0, 0, 0) * rate *
				(comp1[Species::He] + comp2[Species::He]);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][1][0][0]), f * cR2);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][1][1][0]), f * cR1);

			// Trap mutation case
			if (this->_products[1] != Superclass::invalidIndex) {
				// The second product increases
				f = this->_coefs(0, 0, 0, 0) * rate;
				Kokkos::atomic_add(
					&values(this->_connEntries[3][0][0][0]), f * cR2);
				Kokkos::atomic_add(
					&values(this->_connEntries[3][0][1][0]), f * cR1);

				Composition prodComp(
					this->_clusterData->getCluster(this->_products[1])
						.getRegion()
						.getOrigin());
				f = this->_coefs(0, 0, 0, 0) * rate *
					(comp1[Species::V] + comp2[Species::V] +
						prodComp[Species::I]);
				Kokkos::atomic_add(
					&values(this->_connEntries[2][2][0][0]), f * cR2);
				Kokkos::atomic_add(
					&values(this->_connEntries[2][2][1][0]), f * cR1);
			}
			else {
				f = this->_coefs(0, 0, 0, 0) * rate *
					(comp1[Species::V] + comp2[Species::V]);
				Kokkos::atomic_add(
					&values(this->_connEntries[2][2][0][0]), f * cR2);
				Kokkos::atomic_add(
					&values(this->_connEntries[2][2][1][0]), f * cR1);
			}

			return;
		}
		// V case
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
			// The large bubble increases, as well as average He and V
			Kokkos::atomic_add(
				&values(this->_connEntries[2][0][0][0]), f * cR2);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][0][1][0]), f * cR1);
			f = this->_coefs(0, 0, 0, 0) * rate *
				(comp1[Species::He] + comp2[Species::He]);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][1][0][0]), f * cR2);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][1][1][0]), f * cR1);
			f = this->_coefs(0, 0, 0, 0) * rate *
				(comp1[Species::V] + comp2[Species::V]);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][2][0][0]), f * cR2);
			Kokkos::atomic_add(
				&values(this->_connEntries[2][2][1][0]), f * cR1);

			return;
		}
	}
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
double
PSIProductionReaction<TSpeciesEnum>::getRateForProduction(IndexType gridIndex)
{
	auto cl0 = this->_clusterData->getCluster(this->_reactants[0]);
	auto cl1 = this->_clusterData->getCluster(this->_reactants[1]);

	double temperature = cl0.getTemperature(gridIndex);
	double atomicVolume = this->_clusterData->atomicVolume();

	// Standard case
	if (not isLargeBubbleReaction) {
		double r0 = cl0.getReactionRadius();
		double r1 = cl1.getReactionRadius();

		double dc0 = cl0.getDiffusionCoefficient(gridIndex);
		double dc1 = cl1.getDiffusionCoefficient(gridIndex);

		return psi::getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1,
			temperature, atomicVolume);
	}

	// static
	const auto dummyRegion = Region(Composition{});

	double r0 = 0.0, r1 = 0.0, dc0 = 0.0, dc1 = 0.0;
	Region cl0Reg = dummyRegion, cl1Reg = dummyRegion;
	auto largeBubbleId = this->_clusterData->bubbleId();

	if (this->_reactants[0] == largeBubbleId) {
		r0 = this->_clusterData->bubbleAvRadius();
		cl0Reg[Species::He] = {1, 2};
		cl0Reg[Species::V] = {1, 2};
	}
	else {
		auto cl0 = this->_clusterData->getCluster(this->_reactants[0]);
		r0 = cl0.getReactionRadius();
		dc0 = cl0.getDiffusionCoefficient(gridIndex);
		cl0Reg = cl0.getRegion();
	}

	if (this->_reactants[1] == largeBubbleId) {
		r1 = this->_clusterData->bubbleAvRadius();
		cl1Reg[Species::He] = {1, 2};
		cl1Reg[Species::V] = {1, 2};
	}
	else {
		auto cl1 = this->_clusterData->getCluster(this->_reactants[1]);
		r1 = cl1.getReactionRadius();
		dc1 = cl1.getDiffusionCoefficient(gridIndex);
		cl1Reg = cl1.getRegion();
	}

	return psi::getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1,
		temperature, atomicVolume);
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
double
PSIDissociationReaction<TSpeciesEnum>::getRateForProduction(IndexType gridIndex)
{
	auto cl0 = this->_clusterData->getCluster(this->_products[0]);
	auto cl1 = this->_clusterData->getCluster(this->_products[1]);

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	double dc0 = cl0.getDiffusionCoefficient(gridIndex);
	double dc1 = cl1.getDiffusionCoefficient(gridIndex);

	double temperature = cl0.getTemperature(gridIndex);
	double atomicVolume = this->_clusterData->atomicVolume();

	return psi::getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1,
		temperature, atomicVolume);
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
double
PSIDissociationReaction<TSpeciesEnum>::computeBindingEnergy(double time)
{
	using psi::hasDeuterium;
	using psi::hasTritium;

	using NetworkType = typename Superclass::NetworkType;

	constexpr double beTableV1[10][7] = {
		// H:  1     2     3     4     5     6      // He:
		{0.0, 1.21, 1.17, 1.05, 0.93, 0.85, 0.60}, // 0
		{0.0, 1.00, 0.95, 0.90, 0.88, 0.80, 0.60}, // 1
		{0.0, 0.96, 0.92, 0.85, 0.84, 0.83, 0.50}, // 2
		{0.0, 0.86, 0.81, 0.69, 0.64, 0.65, 0.50}, // 3
		{0.0, 0.83, 0.80, 0.65, 0.60, 0.60, 0.55}, // 4
		{0.0, 0.83, 0.80, 0.60, 0.50, 0.50, 0.50}, // 5
		{0.0, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50}, // 6
		{0.0, 0.80, 0.75, 0.65, 0.55, 0.55, 0.45}, // 7
		{0.0, 0.80, 0.80, 0.70, 0.65, 0.60, 0.55}, // 8
		{0.0, 0.80, 0.80, 0.75, 0.70, 0.65, 0.60}, // 9
	};

	constexpr double beTableV2[15][12] = {
		// H:  1     2     3     4     5     6     7     8     9     10    11 //
		// He:
		{0.0, 1.63, 1.31, 1.25, 1.16, 1.00, 1.00, 0.95, 0.95, 0.75, 0.70,
			0.65}, // 0
		{0.0, 1.30, 1.30, 1.24, 1.08, 0.95, 0.95, 0.95, 0.95, 0.75, 0.70,
			0.65}, // 1
		{0.0, 1.15, 1.14, 1.11, 1.14, 0.95, 0.95, 0.95, 0.90, 0.75, 0.70,
			0.65}, // 2
		{0.0, 1.12, 1.06, 0.99, 0.99, 0.90, 0.95, 0.90, 0.90, 0.70, 0.70,
			0.65}, // 3
		{0.0, 1.10, 1.06, 0.99, 0.99, 0.90, 0.95, 0.90, 0.90, 0.70, 0.65,
			0.65}, // 4
		{0.0, 1.10, 1.05, 0.99, 0.99, 0.90, 0.90, 0.90, 0.90, 0.70, 0.65,
			0.65}, // 5
		{0.0, 1.10, 1.05, 0.99, 0.99, 0.90, 0.90, 0.90, 0.85, 0.70, 0.65,
			0.60}, // 6
		{0.0, 1.05, 1.00, 0.95, 0.95, 0.90, 0.90, 0.90, 0.85, 0.65, 0.65,
			0.60}, // 7
		{0.0, 1.05, 1.00, 0.95, 0.95, 0.90, 0.90, 0.85, 0.85, 0.65, 0.65,
			0.60}, // 8
		{0.0, 1.05, 1.00, 0.95, 0.95, 0.85, 0.85, 0.85, 0.85, 0.65, 0.65,
			0.60}, // 9
		{0.0, 1.00, 0.95, 0.90, 0.90, 0.85, 0.85, 0.85, 0.80, 0.65, 0.60,
			0.60}, // 10
		{0.0, 0.95, 0.95, 0.90, 0.90, 0.85, 0.85, 0.85, 0.80, 0.65, 0.60,
			0.60}, // 11
		{0.0, 0.95, 0.90, 0.90, 0.85, 0.85, 0.85, 0.80, 0.80, 0.60, 0.60,
			0.55}, // 12
		{0.0, 0.90, 0.90, 0.85, 0.85, 0.85, 0.85, 0.80, 0.80, 0.60, 0.60,
			0.55}, // 13
		{0.0, 0.90, 0.90, 0.85, 0.85, 0.80, 0.80, 0.80, 0.70, 0.60, 0.60,
			0.55}, // 14
	};

	using Species = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;
	using AmountType = typename NetworkType::AmountType;

	double be = 0.0;

	auto cl = this->_clusterData->getCluster(this->_reactant);
	auto prod1 = this->_clusterData->getCluster(this->_products[0]);
	auto prod2 = this->_clusterData->getCluster(this->_products[1]);

	auto clReg = cl.getRegion();
	auto prod1Reg = prod1.getRegion();
	auto prod2Reg = prod2.getRegion();
	bool useTable = false;
	if (clReg.isSimplex()) {
		if (prod1Reg.isSimplex()) {
			auto orig1 = prod1Reg.getOrigin();
			if constexpr (hasDeuterium<Species> && hasTritium<Species>) {
				if (orig1.isOnAxis(Species::D) || orig1.isOnAxis(Species::T)) {
					useTable = true;
				}
			}
		}
		else if (prod2Reg.isSimplex()) {
			auto orig2 = prod2Reg.getOrigin();
			if constexpr (hasDeuterium<Species> && hasTritium<Species>) {
				if (orig2.isOnAxis(Species::D) || orig2.isOnAxis(Species::T)) {
					useTable = true;
				}
			}
		}
	}

	if constexpr (hasDeuterium<Species> && hasTritium<Species>) {
		if (useTable) {
			Composition comp(clReg.getOrigin());
			auto hAmount = comp[Species::D] + comp[Species::T];
			if (comp[Species::V] == 1) {
				be = beTableV1[comp[Species::He]][hAmount];
			}
			else if (comp[Species::V] == 2) {
				be = beTableV2[comp[Species::He]][hAmount];
			}
		}
	}

	if (be == 0.0) {
		// Special case for V
		auto orig1 = prod1Reg.getOrigin();
		auto orig2 = prod2Reg.getOrigin();
		Composition comp(clReg.getOrigin());
		AmountType lowerV = 16, higherV = 31;
		AmountType minV = 1;

		for (auto i = 1; i < higherV; i++) {
			auto maxHe = util::getMaxHePerVLoop(i,
				this->_clusterData->latticeParameter(), cl.getTemperature(0));
			if (comp[Species::He] > maxHe)
				minV = i;
		}
		lowerV = util::max(lowerV, minV + 2);
		if ((orig1.isOnAxis(Species::V) || orig2.isOnAxis(Species::V)) &&
			(comp[Species::V] >= lowerV && comp[Species::V] <= higherV)) {
			// Get the be at 16 and 30
			Composition HeVComp(clReg.getOrigin());
			HeVComp[Species::V] = lowerV;
			auto fe1 = PSIClusterGenerator<TSpeciesEnum>::getHeVFormationEnergy(
				HeVComp);
			HeVComp[Species::V] = lowerV - 1;
			auto fe2 = PSIClusterGenerator<TSpeciesEnum>::getHeVFormationEnergy(
				HeVComp);
			Composition vComp{};
			vComp[Species::V] = 1;
			auto fe3 =
				PSIClusterGenerator<TSpeciesEnum>::getHeVFormationEnergy(vComp);
			auto be1 = fe2 + fe3 - fe1;
			HeVComp[Species::V] = higherV;
			fe1 = PSIClusterGenerator<TSpeciesEnum>::getHeVFormationEnergy(
				HeVComp);
			HeVComp[Species::V] = higherV - 1;
			fe2 = PSIClusterGenerator<TSpeciesEnum>::getHeVFormationEnergy(
				HeVComp);
			auto be2 = fe2 + fe3 - fe1;
			if (higherV - lowerV < 4)
				be = be2;
			else
				be = be1 +
					(comp[Species::V] - lowerV) * (be2 - be1) /
						(higherV - lowerV);
		}
		else {
			be = prod1.getFormationEnergy() + prod2.getFormationEnergy() -
				cl.getFormationEnergy();
		}
	}

	return util::max(be, -5.0);
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
double
PSISinkReaction<TSpeciesEnum>::getSinkBias()
{
	return 1.0;
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
double
PSISinkReaction<TSpeciesEnum>::getSinkStrength()
{
	constexpr double pi = ::xolotl::core::pi;
	double grainSize = 50000.0; // 50 um

	return 1.0 / (pi * grainSize * grainSize);
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
double
PSIBurstingReaction<TSpeciesEnum>::getAppliedRate(IndexType gridIndex) const
{
	using NetworkType = typename Superclass::NetworkType;
	using Species = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;
	using AmountType = typename NetworkType::AmountType;

	// Get the radius of the cluster
	double radius = 0.0;
	if (isLargeBubbleReaction) {
		radius = this->_clusterData->bubbleAvRadius();
	}
	else {
		auto cl = this->_clusterData->getCluster(this->_reactant);
		radius = cl.getReactionRadius();
	}

	// Get the current depth
	auto depth = this->_clusterData->getDepth();
	auto tau = this->_clusterData->getTauBursting();
	auto f = this->_clusterData->getFBursting();

	return f * (radius / depth) *
		util::min(1.0, exp(-(depth - tau) / (2.0 * tau)));
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
void
PSIBurstingReaction<TSpeciesEnum>::computeCoefficients()
{
	// Check if the large bubble is involved
	if (isLargeBubbleReaction) {
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

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
void
PSIBurstingReaction<TSpeciesEnum>::computeFlux(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex)
{
	// Standard case
	if (not isLargeBubbleReaction) {
		return Superclass::computeFlux(concentrations, fluxes, gridIndex);
	}

	auto rate = this->getAppliedRate(gridIndex);

	Kokkos::atomic_sub(&fluxes[this->_clusterData->bubbleAvHeId()],
		rate * concentrations[this->_clusterData->bubbleAvHeId()]);
}

template <typename TSpeciesEnum>
KOKKOS_INLINE_FUNCTION
void
PSIBurstingReaction<TSpeciesEnum>::computePartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	IndexType gridIndex)
{
	// Standard case
	if (not isLargeBubbleReaction) {
		return Superclass::computePartialDerivatives(
			concentrations, values, gridIndex);
	}

	auto rate = this->getAppliedRate(gridIndex);

	Kokkos::atomic_sub(&values(this->_connEntries[0][1][0][1]), rate);
}
} // namespace network
} // namespace core
} // namespace xolotl
