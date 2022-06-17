#include "../../include/raster/mesh.hpp"
#include "../../include/raster/layer.hpp"

namespace kobra {

namespace raster {

// Latch to layer
void Mesh::latch(const LatchingPacket &lp)
{
	// Always get the local descriptor set
	_dset = lp.layer->serve_ds();

	_material.bind(lp.phdev, lp.device, lp.command_pool,
		_dset,
		RASTER_BINDING_ALBEDO_MAP,
		RASTER_BINDING_NORMAL_MAP
	);
}

}

}
