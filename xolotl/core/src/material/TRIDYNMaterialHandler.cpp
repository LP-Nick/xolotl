#include <xolotl/core/material/TRIDYNMaterialHandler.h>

namespace xolotl
{
namespace core
{
namespace material
{
namespace detail
{
auto tridynMaterialHandlerRegistrations =
	xolotl::factory::material::MaterialHandlerFactory::RegistrationCollection<
		TRIDYNMaterialHandler>({"TRIDYN"});
}
} // namespace material
} // namespace core
} // namespace xolotl
