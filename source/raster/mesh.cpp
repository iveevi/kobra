#include "../../include/raster/mesh.hpp"
#include "../../include/raster/layer.hpp"

namespace kobra {

namespace raster {

// Latch to layer
void Mesh::latch(const LatchingPacket &lp)
{
	// Always get the local descriptor set
	_ds = lp.layer->serve_ds();

	_material.bind(*lp.context,
		*lp.command_pool, _ds,
		RASTER_BINDING_ALBEDO_MAP,
		RASTER_BINDING_NORMAL_MAP
	);

	// Only do stuff if the mesh is emissive
	if (_material.shading_type != SHADING_TYPE_EMISSIVE)
		return;

	glm::vec3 pos = center();
	lp.ubo_point_lights->positions
		[lp.ubo_point_lights->number++] = pos;
}

}

}
