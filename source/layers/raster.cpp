// Engine headers
#include "../../include/layers/raster.hpp"
#include "../../shaders/raster/bindings.h"

namespace kobra {

namespace layers {

// Static variables
const std::vector <DSLB> Raster::_dsl_bindings {
	DSLB {
		RASTER_BINDING_ALBEDO_MAP,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_NORMAL_MAP,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_POINT_LIGHTS,
		vk::DescriptorType::eUniformBuffer,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

}

}
