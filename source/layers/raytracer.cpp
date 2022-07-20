// Engine headers
#include "../../include/layers/raytracer.hpp"

namespace kobra {

namespace layers {

const std::vector <DSLB> Raytracer::_dslb_raytracing {
	DSLB {
		MESH_BINDING_PIXELS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_VERTICES,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_TRIANGLES,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_TRANSFORMS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_BVH,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Materials buffer
	DSLB {
		MESH_BINDING_MATERIALS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Lights buffer
	DSLB {
		MESH_BINDING_LIGHTS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Light indices
	DSLB {
		MESH_BINDING_LIGHT_INDICES,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Texture samplers
	DSLB {
		MESH_BINDING_ALBEDOS,
		vk::DescriptorType::eCombinedImageSampler,
		MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_NORMAL_MAPS,
		vk::DescriptorType::eCombinedImageSampler,
		MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_ENVIRONMENT,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_OUTPUT,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},
};

const std::vector <DSLB> Raytracer::_dslb_postprocess = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

}

}
