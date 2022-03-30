// Engine headers
#include "../../include/raytracing/layer.hpp"
#include "../../shaders/rt/mesh_bindings.h"

namespace kobra {

namespace rt {

/////////////////////////////
// Static member variables //
/////////////////////////////

const Layer::DSLBindings Layer::_mesh_compute_bindings {
	DSLBinding {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = MESH_BINDING_PIXELS,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = MESH_BINDING_VIEWPORT,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	}
};

}

}
