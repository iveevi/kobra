#ifndef RASTER_H_
#define RASTER_H_

// Engine headers
#include "../buffer_manager.hpp"
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
};

// Rasterization elements
struct _element {
	// Virtual destructor
	virtual ~_element() = default;

	// Virtual methods
	virtual void render(RenderPacket &) = 0;
};

// Shared pointer alias
using Element = std::shared_ptr <_element>;

}

}

#endif
