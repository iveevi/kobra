#include "../../include/raster/layer.hpp"
#include "../../include/raster/mesh.hpp"
#include "../../include/texture_manager.hpp"

namespace kobra {

namespace raster {

// Latch to layer
void Mesh::latch(const LatchingPacket &lp)
{
	// Always get the local descriptor set
	_dset = lp.layer->serve_ds();

	std::string albedo = "blank";
	if (material.has_albedo())
		albedo = material.albedo_source;

	std::string normal = "blank";
	if (material.has_normal())
		normal = material.normal_source;

	TextureManager::bind(
		lp.phdev, lp.device,
		_dset, albedo,
		RASTER_BINDING_ALBEDO_MAP
	);

	TextureManager::bind(
		lp.phdev, lp.device,
		_dset, normal,
		RASTER_BINDING_NORMAL_MAP
	);

}

}

}
