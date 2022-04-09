#include "../../include/raster/mesh.hpp"
#include "../../include/raster/layer.hpp"

namespace kobra {

namespace raster {

// Latch to layer
void Mesh::latch(const LatchingPacket &lp)
{
	// Always get the local descriptor set
	_ds = lp.layer->serve_ds();

	// Configure albedo
	if (_material.albedo_sampler) {
		_material.albedo_sampler->bind(_ds, RASTER_BINDING_ALBEDO_MAP);
	} else {
		Sampler blank = Sampler::blank_sampler(*lp.context, *lp.command_pool);
		blank.bind(_ds, RASTER_BINDING_ALBEDO_MAP);
	}

	// Configure normal
	if (_material.normal_sampler) {
		_material.normal_sampler->bind(_ds, RASTER_BINDING_NORMAL_MAP);
	} else {
		Sampler blank = Sampler::blank_sampler(*lp.context, *lp.command_pool);
		blank.bind(_ds, RASTER_BINDING_NORMAL_MAP);
	}

	// Only do stuff if the mesh is emissive
	if (_material.shading_type != SHADING_TYPE_EMISSIVE)
		return;

	glm::vec3 pos = _transform.apply(centroid());
	lp.ubo_point_lights->positions
		[lp.ubo_point_lights->number++] = pos;
}

}

}
