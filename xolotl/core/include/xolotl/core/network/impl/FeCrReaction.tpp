#pragma once

#include <xolotl/core/network/impl/SinkReaction.tpp>
#include <xolotl/core/network/impl/TransformReaction.tpp>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
template <typename TRegion>
KOKKOS_INLINE_FUNCTION
double
getRate(const TRegion& pairCl0Reg, const TRegion& pairCl1Reg, const double r0,
	const double r1, const double dc0, const double dc1, const double rho,
	const double sigma0, const double sigma1)
{
	constexpr double pi = ::xolotl::core::pi;
	double zs = 4.0 * pi * (r0 + r1 + ::xolotl::core::fecrCoreRadius);

	using Species = typename TRegion::EnumIndex;
	auto lo0 = pairCl0Reg.getOrigin();
	auto lo1 = pairCl1Reg.getOrigin();

	// react_extension
	if ((lo0.isOnAxis(Species::I) and lo1.isOnAxis(Species::V)) or
		(lo0.isOnAxis(Species::V) and lo1.isOnAxis(Species::I)))
		return 4.0 * pi * (r0 + r1 + ::xolotl::core::fecrCoreRadius + 0.25) *
			(dc0 + dc1);

	bool cl0IsSphere = (lo0.isOnAxis(Species::V) ||
			 lo0.isOnAxis(Species::Complex) || lo0.isOnAxis(Species::I)),
		 cl1IsSphere = (lo1.isOnAxis(Species::V) ||
			 lo1.isOnAxis(Species::Complex) || lo1.isOnAxis(Species::I));
	bool cl0IsTrap = lo0.isOnAxis(Species::Trap),
		 cl1IsTrap = lo1.isOnAxis(Species::Trap);

	// Simple case
	if (cl0IsSphere && cl1IsSphere) {
		return zs * (dc0 + dc1);
	}

	// Sphere and trap
	if ((cl0IsSphere and cl1IsTrap) or (cl1IsSphere and cl0IsTrap)) {
		return zs * (dc0 + dc1);
	}

	// With a trap
	if (cl0IsTrap || cl1IsTrap) {
		double rT = cl0IsTrap ? r0 : r1;
		double rL = cl0IsTrap ? r1 : r0;
		double sigmaL = cl0IsTrap ? sigma1 : sigma0;
		double sigma = pi * (r0 + r1) * (r0 + r1);
		if (rT < rL)
			sigma = 4.0 * pi * r0 * r1;
		return 2.0 * (dc0 + dc1) * sigma * sigmaL;
	}

	double rB = (r0 > r1) ? r0 : r1;
	double rs = (r0 > r1) ? r1 : r0;
	double rint = ::xolotl::core::fecrCoreRadius + rs;
	double ratio = rB / (3.0 * rint);
	double p = 1.0 / (1.0 + ratio * ratio);
	double zd = 4.0 * pi * pi * rB / log(1.0 + 8.0 * rB / rint);

	// Minimum loop size
	rB = (r0 > r1) ? r0 : r1;
	double zd_alt = 0.0;
	if (rho > 0.0)
		zd_alt = -8.0 * pi * pi * rB / log(pi * rho * rint * rint);

	// Loop + loop
	if (!cl0IsSphere and !cl1IsSphere) {
		double sigmaL = dc0 > 0.0 ? sigma0 : sigma1;
		double rSessile = dc0 > 0.0 ? r1 : r0;
		double rFree = dc0 > 0.0 ? r0 : r1;
		double dFree = dc0 > 0.0 ? dc0 : dc1;

		util::Array<double, 3, 2> align;
		align[0] = {{0.0, 0.0}};
		align[1] = {{0.0, 0.0}};
		align[2] = {{0.0, 0.0}};

		// Find what type of cluster is the sessile one
		auto loSessile = dc0 > 0.0 ? lo1 : lo0;
		if (loSessile[static_cast<int>(Species::Trapped)] > 0) {
			align[0] = {{1.0, 0.25}};
			align[1] = {{0.333333, 0.75}};
		}
		else if (loSessile[static_cast<int>(Species::Junction)] > 0) {
			align[0] = {{1.0, 0.142857}};
			align[1] = {{0.333333, 0.4285714}};
			align[2] = {{0.57735, 0.4285714}};
		}
		else if (loSessile[static_cast<int>(Species::Loop)] > 0) {
			align[0] = {{0.57735, 1.0}};
		}

		double sigma = 0.0;
		for (auto i = 0; i < align.size(); i++) {
			double ao = rFree + rSessile + ::xolotl::core::fecrCoalesceRadius;
			double bo = rFree + rSessile * align[i][0] +
				::xolotl::core::fecrCoalesceRadius;
			double ai = util::max(
				rFree + rSessile - ::xolotl::core::fecrCoalesceRadius, 0.0);
			double bi = util::max(rFree + rSessile * align[i][0] -
					::xolotl::core::fecrCoalesceRadius,
				0.0);

			sigma += pi * (ao * bo - ai * bi) * align[i][1];
		}
		double k_plus = 2.0 * dFree * sigma * sigmaL;

		return k_plus;
	}

	double dsphere = cl0IsSphere ? dc0 : dc1;

	double k_plus = dsphere * (p * zs + (1.0 - p) * util::max(zd, zd_alt));

	// Loop + Sphere point cross section
	if (!cl0IsSphere or !cl1IsSphere) {
		double sigmaL = cl0IsSphere ? sigma1 : sigma0;
		double rsphere = cl0IsSphere ? r0 : r1;
		double rloop = cl0IsSphere ? r1 : r0;
		double dloop = cl0IsSphere ? dc1 : dc0;
		rint = rsphere + ::xolotl::core::fecrCoreRadius;
		double sigma = pi * (rloop + rint) * (rloop + rint);
		if (rloop > rint)
			sigma = 4.0 * pi * rloop * rint;
		k_plus += 2.0 * dloop * sigma * sigmaL;
	}

	double bias = 1.0;
	if (lo0.isOnAxis(Species::I) || lo1.isOnAxis(Species::I)) {
		if (lo0.isOnAxis(Species::Free) || lo1.isOnAxis(Species::Free))
			bias = 1.0;
		else if (lo0[static_cast<int>(Species::Loop)] > 0 ||
			lo1[static_cast<int>(Species::Loop)] > 0)
			bias = 1.05774;
		else
			bias = 1.05;
	}

	return k_plus * bias;
}

KOKKOS_INLINE_FUNCTION
double
FeCrProductionReaction::getRateForProduction(IndexType gridIndex)
{
	auto cl0 = this->_clusterData->getCluster(_reactants[0]);
	auto cl1 = this->_clusterData->getCluster(_reactants[1]);

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	double dc0 = cl0.getDiffusionCoefficient(gridIndex);
	double dc1 = cl1.getDiffusionCoefficient(gridIndex);

	auto rho = this->_clusterData->sinkDensity(); // nm-2

	return getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1, rho,
		this->_clusterData->extraData.netSigma(_reactants[0]),
		this->_clusterData->extraData.netSigma(_reactants[1]));
}

KOKKOS_INLINE_FUNCTION
double
FeCrProductionReaction::computeNetSigma(
	ConcentrationsView concentrations, IndexType clusterId, IndexType gridIndex)
{
	double preFactor = 0.0;
	// Check if our cluster is on the left side of this reaction
	if (clusterId == _reactants[0]) {
		preFactor = concentrations[_reactants[1]] * this->_coefs(0, 0, 0, 0);
	}
	if (clusterId == _reactants[1]) {
		preFactor = concentrations[_reactants[0]] * this->_coefs(0, 0, 0, 0);
	}

	if (clusterId == _reactants[0] or clusterId == _reactants[1]) {
		// Compute cross section
		auto cl0 = this->_clusterData->getCluster(_reactants[0]);
		auto cl1 = this->_clusterData->getCluster(_reactants[1]);

		double r0 = cl0.getReactionRadius();
		double r1 = cl1.getReactionRadius();
		auto lo0 = cl0.getRegion().getOrigin();
		auto lo1 = cl1.getRegion().getOrigin();

		bool cl0IsSphere = (lo0.isOnAxis(Species::V) ||
				 lo0.isOnAxis(Species::Complex) || lo0.isOnAxis(Species::I)),
			 cl1IsSphere = (lo1.isOnAxis(Species::V) ||
				 lo1.isOnAxis(Species::Complex) || lo1.isOnAxis(Species::I));
		bool cl0IsTrap = lo0.isOnAxis(Species::Trap),
			 cl1IsTrap = lo1.isOnAxis(Species::Trap);
		// Sphere + sphere
		if (cl0IsSphere && cl1IsSphere) {
			return 0.0;
		}

		// With a trap
		if (cl0IsTrap || cl1IsTrap) {
			double rT = cl0IsTrap ? r0 : r1;
			double rL = cl0IsTrap ? r1 : r0;
			double sigma = pi * (r0 + r1) * (r0 + r1);
			if (rT < rL)
				sigma = 4.0 * pi * r0 * r1;
			return preFactor * sigma;
		}

		// Loop + loop
		if (!cl0IsSphere and !cl1IsSphere) {
			return 0.0;
		}

		// Loop + sphere
		if (!cl0IsSphere or !cl1IsSphere) {
			double rsphere = cl0IsSphere ? r0 : r1;
			double rloop = cl0IsSphere ? r1 : r0;
			double rint = rsphere + ::xolotl::core::fecrCoreRadius;
			double sigma = pi * (rloop + rint) * (rloop + rint);
			if (rloop > rint)
				sigma = 4.0 * pi * rloop * rint;
			return preFactor * sigma;
		}
	}

	// This cluster is not part of the reaction
	return 0.0;
}

KOKKOS_INLINE_FUNCTION
double
FeCrDissociationReaction::getRateForProduction(IndexType gridIndex)
{
	auto cl0 = this->_clusterData->getCluster(_products[0]);
	auto cl1 = this->_clusterData->getCluster(_products[1]);

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	double dc0 = cl0.getDiffusionCoefficient(gridIndex);
	double dc1 = cl1.getDiffusionCoefficient(gridIndex);

	auto rho = this->_clusterData->sinkDensity(); // nm-2

	double kPlus = getRate(cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1,
		rho, this->_clusterData->extraData.netSigma(_products[0]),
		this->_clusterData->extraData.netSigma(_products[1]));

	return kPlus;
}

KOKKOS_INLINE_FUNCTION
double
FeCrDissociationReaction::computeRate(IndexType gridIndex, double)
{
	double T = this->_clusterData->temperature(gridIndex);
	constexpr double pi = ::xolotl::core::pi;
	using Species = typename Superclass::Species;
	using Composition = typename Superclass::Composition;

	double kPlus = this->asDerived()->getRateForProduction(gridIndex);
	double E_b = this->asDerived()->computeBindingEnergy();

	constexpr double k_B = ::xolotl::core::kBoltzmann;

	auto cl0 = this->_clusterData->getCluster(_products[0]);
	auto cl1 = this->_clusterData->getCluster(_products[1]);

	auto lo = this->_clusterData->getCluster(_reactant).getRegion().getOrigin();
	auto lo0 = cl0.getRegion().getOrigin();
	auto lo1 = cl1.getRegion().getOrigin();
	bool cl0IsSphere = (lo0.isOnAxis(Species::V) ||
			 lo0.isOnAxis(Species::Complex) || lo0.isOnAxis(Species::I)),
		 cl1IsSphere = (lo1.isOnAxis(Species::V) ||
			 lo1.isOnAxis(Species::Complex) || lo1.isOnAxis(Species::I));
	bool cl0IsTrap = lo0.isOnAxis(Species::Trap),
		 cl1IsTrap = lo1.isOnAxis(Species::Trap);

	double kMinus = kPlus * std::exp(-E_b / (k_B * T));

	// Standard case
	if (cl0IsSphere and cl1IsSphere) {
		return (kMinus / this->_clusterData->atomicVolume());
	}

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	// Sphere and trap
	if ((cl0IsSphere and cl1IsTrap) or (cl1IsSphere and cl0IsTrap)) {
		return (kMinus / this->_clusterData->atomicVolume());
	}

	double rsphere = cl0IsSphere ? r0 : r1;
	double rloop = cl0IsSphere ? r1 : r0;

	double rint = rsphere + ::xolotl::core::fecrCoreRadius;
	double sigma = pi * (rloop + rint) * (rloop + rint);
	if (rloop > rint)
		sigma = 4.0 * pi * rloop * rint;

	if (cl0IsTrap or cl1IsTrap) {
		double rT = cl0IsTrap ? r0 : r1;
		double rL = cl0IsTrap ? r1 : r0;
		sigma = pi * (r0 + r1) * (r0 + r1);
		if (rT < rL)
			sigma = 4.0 * pi * r0 * r1;
	}

	double preFactor = 1.0 /
		(sigma * this->_clusterData->latticeParameter() *
			::xolotl::core::fecrBurgers);

	return kMinus * preFactor;
}

KOKKOS_INLINE_FUNCTION
double
FeCrDissociationReaction::computeBindingEnergy(double time)
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
	if (lo.isOnAxis(Species::V)) {
		double n = (double)(lo[Species::V] + hi[Species::V] - 1) / 2.0;
		if (prod1Comp.isOnAxis(Species::V) && prod2Comp.isOnAxis(Species::V)) {
			be = 1.73 - 2.59 * (pow(n, 2.0 / 3.0) - pow(n - 1.0, 2.0 / 3.0));
		}
	}
	else if (lo[Species::Complex] > 0) {
		be = 0.01;
	}
	else if (lo.isOnAxis(Species::I)) {
		double n = (double)(lo[Species::I] + hi[Species::I] - 1) / 2.0;
		if (prod1Comp.isOnAxis(Species::I) && prod2Comp.isOnAxis(Species::I)) {
			be = 4.33 - 5.76 * (pow(n, 2.0 / 3.0) - pow(n - 1.0, 2.0 / 3.0));
		}
	}
	else if (lo.isOnAxis(Species::Free)) {
		double n = (double)(lo[Species::Free] + hi[Species::Free] - 1) / 2.0;
		be = 4.33 - 5.76 * (pow(n, 2.0 / 3.0) - pow(n - 1.0, 2.0 / 3.0));
	}
	else if (lo[Species::Trapped] > 0) {
		if (prod1Comp.isOnAxis(Species::I) || prod2Comp.isOnAxis(Species::I)) {
			double n =
				(double)(lo[Species::Trapped] + hi[Species::Trapped] - 1) / 2.0;
			be = 4.33 - 5.76 * (pow(n, 2.0 / 3.0) - pow(n - 1.0, 2.0 / 3.0));
		}
		else if (prod1Comp.isOnAxis(Species::Trap) ||
			prod2Comp.isOnAxis(Species::Trap)) {
			be = 1.2;
		}
	}
	else if (lo[Species::Junction] > 0) {
		be = 2.5;
	}

	return util::min(5.0, util::max(be, 0.1));
}

KOKKOS_INLINE_FUNCTION
double
FeCrSinkReaction::computeRate(IndexType gridIndex, double time)
{
	auto cl = this->_clusterData->getCluster(_reactant);
	double dc = cl.getDiffusionCoefficient(gridIndex);

	double strength = this->asDerived()->getSinkStrength() * dc;

	auto clReg = cl.getRegion();
	Composition comp = clReg.getOrigin();
	if (comp.isOnAxis(Species::Free)) {
		strength *= this->_clusterData->extraData.netSigma(this->_reactant);
	}

	return strength;
}

KOKKOS_INLINE_FUNCTION
double
FeCrSinkReaction::getSinkBias()
{
	using Species = typename Superclass::Species;
	using Composition = typename Superclass::Composition;

	double bias = 1.0;

	auto cl = this->_clusterData->getCluster(this->_reactant);

	auto clReg = cl.getRegion();
	if (clReg.isSimplex()) {
		Composition comp = clReg.getOrigin();
		if (comp.isOnAxis(Species::I)) {
			bias = 1.05;
		}
	}

	return bias;
}

KOKKOS_INLINE_FUNCTION
double
FeCrSinkReaction::getSinkStrength()
{
	auto density = this->_clusterData->sinkDensity(); // nm-2
	auto portion = this->_clusterData->sinkPortion(); // portion of screw
	auto r = 1.0 / sqrt(::xolotl::core::pi * density); // nm
	auto rCore = ::xolotl::core::fecrCoreRadius;
	auto temperature = this->_clusterData->temperature(0);
	constexpr double K = 170.0e9; // GPa
	constexpr double nu = 0.29;
	constexpr double b = 0.25; // nm
	double deltaV = 1.67 * this->_clusterData->atomicVolume() * 1.0e-27; // m3
	//	constexpr double a0 = 0.91, a1 = -2.16, a2 = -0.92; // Random dipole
	constexpr double a0 = 0.87, a1 = -5.12, a2 = -0.77; // Full network
	constexpr double k_B = 1.380649e-23; // J K-1.

	double L = (K * b * deltaV * (1.0 - 2.0 * nu)) /
		(2.0 * ::xolotl::core::pi * k_B * temperature * (1.0 - nu));

	double delta = sqrt(rCore * rCore + (L * L) / 4.0);

	double Z =
		(2.0 * ::xolotl::core::pi * (a0 + a1 * (rCore / r)) *
			(portion +
				(1.0 - portion) * this->getSinkBias() *
					((std::log(r / rCore) *
						 (a0 * r + a1 * delta + a2 * (delta - rCore))) /
						(std::log(r / delta) * (a0 * r + a1 * rCore))))) /
		(std::log(r / rCore));

	return density * Z;
}

KOKKOS_INLINE_FUNCTION
double
FeCrSinkReaction::computeNetSigma(
	ConcentrationsView concentrations, IndexType clusterId, IndexType gridIndex)
{
	// Check if our cluster is on the left side of this reaction
	if (clusterId == _reactant) {
		return this->asDerived()->getSinkStrength();
	}

	// This cluster is not part of the reaction
	return 0.0;
}

KOKKOS_INLINE_FUNCTION
double
FeCrTransformReaction::getSize()
{
	using Species = typename Superclass::Species;
	using Composition = typename Superclass::Composition;

	auto cl = this->_clusterData->getCluster(this->_reactant);

	auto clReg = cl.getRegion();
	Composition comp = clReg.getOrigin();
	return comp[Species::Junction];

	return 0.0;
}

KOKKOS_INLINE_FUNCTION
double
FeCrTransformReaction::getExponent()
{
	// Same for Loop and Trapped

	return 2.0;
}

KOKKOS_INLINE_FUNCTION
double
FeCrTransformReaction::getBarrier()
{
	using Species = typename Superclass::Species;
	using Composition = typename Superclass::Composition;

	auto cl = this->_clusterData->getCluster(this->_product);

	auto clReg = cl.getRegion();
	Composition comp = clReg.getOrigin();
	if (comp[Species::Loop] > 0)
		return this->_clusterData->barrierEnergy(); // Loop
	return 0.75; // Trapped

	return 0.0;
}
} // namespace network
} // namespace core
} // namespace xolotl
