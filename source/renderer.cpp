// Engine headers
#include "../include/renderer.hpp"
#include "../include/texture_manager.hpp"
#include "../shaders/raster/bindings.h"

namespace kobra {

// Rasterizer
void Rasterizer::bind_material(const Device &dev, const vk::raii::DescriptorSet &dset) const
{
	std::string albedo = "blank";
	if (material.has_albedo())
		albedo = material.albedo_source;

	std::string normal = "blank";
	if (material.has_normal())
		normal = material.normal_source;

	TextureManager::bind(
		dev.phdev, dev.device,
		dset, albedo,
		// TODO: enum like RasterBindings::eAlbedo
		RASTER_BINDING_ALBEDO_MAP
	);

	TextureManager::bind(
		dev.phdev, dev.device,
		dset, normal,
		RASTER_BINDING_NORMAL_MAP
	);
}

}
