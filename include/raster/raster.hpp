#ifndef RASTER_H_
#define RASTER_H_

// Engine headers
#include "../../shaders/raster/constants.h"
#include "../object.hpp"
#include "../vertex.hpp"

namespace kobra {

namespace raster {

// Forward declarations
class Layer;

// Rasterization abstraction and primitives
struct RenderPacket {
	const vk::raii::CommandBuffer &cmd;
	const vk::raii::PipelineLayout &pipeline_layout;

	// View and projection matrices
	glm::mat4 view;
	glm::mat4 proj;

	int highlight;
};

//////////////////////
// Light structures //
//////////////////////

// Uniform buffer of point lights
struct UBO_PointLights {
	int number = 0;

	aligned_vec4 positions[MAX_POINT_LIGHTS];
};

// Latching packet
struct LatchingPacket {
	const vk::raii::PhysicalDevice &phdev;
	const vk::raii::Device &device;
	const vk::raii::CommandPool &command_pool;
	Layer		*layer;
};

// Lighting packet
struct LightingPacket {
	UBO_PointLights *ubo_point_lights;
};

// Rasterization elements
struct _element : virtual public Object {
	// Highlight status
	bool highlight = false;

	// Virtual destructor
	virtual ~_element() = default;

	// Virtual methods
	virtual const vk::raii::DescriptorSet &get_local_ds() const = 0;
	virtual void latch(const LatchingPacket &) = 0;
	virtual void light(const LightingPacket &) = 0;
	virtual void render(RenderPacket &) = 0;
};

// Shared pointer alias
using Element = std::shared_ptr <_element>;

}

}

#endif
