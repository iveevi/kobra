#ifndef RASTER_H_
#define RASTER_H_

// Engine headers
#include "../buffer_manager.hpp"
#include "../vertex.hpp"

namespace kobra {

namespace raster {

// More aliases
template <VertexType T>
using VertexBuffer = BufferManager <Vertex <T>>;
using IndexBuffer = BufferManager <uint32_t>;

// Rasterization abstraction and primitives
struct RenderPacket {
	VkCommandBuffer cmd;
};

// Rasterization elements
struct _element {
	// Vertex type
	VertexType type;

	// Virtual methods
	virtual void render(RenderPacket &) = 0;
};

// Shared pointer alias
using Element = std::shared_ptr <_element>;

}

}

#endif
