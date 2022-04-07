#ifndef RASTER_H_
#define RASTER_H_

// Engine headers
#include "../../shaders/raster/constants.h"
#include "../buffer_manager.hpp"
#include "../object.hpp"
#include "../vertex.hpp"

namespace kobra {

namespace raster {

// More aliases
using VertexBuffer = BufferManager <Vertex>;
using IndexBuffer = BufferManager <uint32_t>;

// Rasterization abstraction and primitives
struct RenderPacket {
	VkCommandBuffer cmd;

	VkPipelineLayout pipeline_layout;

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
	UBO_PointLights *ubo_point_lights;
};

// Rasterization elements
struct _element : virtual public Object {
	// Virtual destructor
	virtual ~_element() = default;

	// Latch to layer
	virtual void latch(const LatchingPacket &) = 0;

	// Virtual methods
	virtual void render(RenderPacket &) = 0;
};

// Shared pointer alias
using Element = std::shared_ptr <_element>;

}

}

#endif
