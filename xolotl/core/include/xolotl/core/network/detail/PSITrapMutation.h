#pragma once

#include <memory>

#include <xolotl/core/network/detail/TrapMutationHandler.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace psi
{
class W100TrapMutationHandler : public network::detail::TrapMutationHandler
{
public:
	std::array<std::vector<AmountType>, 7>
	getAllVSizes() const override
	{
		using V = std::vector<AmountType>;
		return {V{}, V{1}, V{1}, V{1}, V{1, 2}, V{2}, V{2}};
	}

	void
	updateData(double temp) override
	{
		// Switch values depending on the temperature
		if (temp < 1066.5) {
			_depths = {-0.1, 0.5, 0.6, 0.6, 0.8, 0.8, 0.8};
			_vSizes = {0, 1, 1, 1, 1, 2, 2};

			// He2 desorpts with 4%
			_desorp.set(2, 0.04);
		}
		else {
			_depths = {-0.1, 0.5, 0.6, 0.8, 0.6, 0.8, 0.8};
			_vSizes = {0, 1, 1, 1, 2, 2, 2};

			// He2 desorpts with 19%
			_desorp.set(2, 0.19);
		}
	}
};

class W110TrapMutationHandler : public network::detail::TrapMutationHandler
{
public:
	std::array<std::vector<AmountType>, 7>
	getAllVSizes() const override
	{
		using V = std::vector<AmountType>;
		return {V{}, V{1}, V{1}, V{1}, V{1}, V{2, 1}, V{2}};
	}

	void
	updateData(double temp) override
	{
		// Switch values depending on the temperature
		if (temp < 1066.5) {
			_depths = {-0.1, 0.7, 0.9, 0.9, 0.9, 1.1, 1.1};
			_vSizes = {0, 1, 1, 1, 1, 2, 2};

			// He2 desorpts with 31%
			_desorp.set(2, 0.31);
		}
		else {
			_depths = {-0.1, 0.7, 0.9, 0.9, 0.9, 1.1, 1.1};
			_vSizes = {0, 1, 1, 1, 1, 1, 2};

			// He2 desorpts with 32%
			_desorp.set(2, 0.32);
		}
	}
};

class W111TrapMutationHandler : public network::detail::TrapMutationHandler
{
public:
	std::array<std::vector<AmountType>, 7>
	getAllVSizes() const override
	{
		using V = std::vector<AmountType>;
		return {V{1}, V{1}, V{1}, V{1}, V{1}, V{1}, V{2}};
	}

	void
	updateData(double temp) override
	{
		// Switch values depending on the temperature
		if (temp < 1066.5) {
			_depths = {0.6, 0.8, 1.1, 1.1, 1.2, 1.3, 1.3};
			_vSizes = {1, 1, 1, 1, 1, 1, 2};

			// He1 desorpts with 61%
			_desorp.set(1, 0.61);
		}
		else {
			_depths = {0.6, 0.8, 1.1, 1.1, 1.1, 1.1, 1.1};
			_vSizes = {1, 1, 1, 1, 1, 1, 2};

			// He1 desorpts with 35%
			_desorp.set(1, 0.35);
		}
	}
};

class W211TrapMutationHandler : public network::detail::TrapMutationHandler
{
public:
	std::array<std::vector<AmountType>, 7>
	getAllVSizes() const override
	{
		using V = std::vector<AmountType>;
		return {V{1}, V{1}, V{1}, V{2}, V{2, 1}, V{2}, V{2, 3}};
	}

	void
	updateData(double temp) override
	{
		// Switch values depending on the temperature
		if (temp < 1066.5) {
			_depths = {0.5, 0.8, 1.0, 1.0, 1.0, 1.3, 1.5};
			_vSizes = {1, 1, 1, 2, 2, 2, 2};

			// He1 desorpts with 64%
			_desorp.set(1, 0.64);
		}
		else {
			_depths = {0.5, 0.8, 1.0, 1.0, 1.3, 1.3, 1.2};
			_vSizes = {1, 1, 1, 2, 1, 2, 3};

			// He1 desorpts with 59%
			_desorp.set(1, 0.59);
		}
	}
};

inline std::unique_ptr<network::detail::TrapMutationHandler>
getTrapMutationHandler(bool enableTrapMutation, const std::string& material)
{
	if (!enableTrapMutation) {
		return nullptr;
	}

	if (material == "W100") {
		return std::make_unique<W100TrapMutationHandler>();
	}
	else if (material == "W110") {
		return std::make_unique<W110TrapMutationHandler>();
	}
	else if (material == "W111") {
		return std::make_unique<W111TrapMutationHandler>();
	}
	else if (material == "W211") {
		return std::make_unique<W211TrapMutationHandler>();
	}
	else {
		throw std::runtime_error("Unsupported PSI trap mutation material");
	}
}
} // namespace psi
} // namespace network
} // namespace core
} // namespace xolotl
