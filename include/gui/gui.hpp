#ifndef GUI_H_
#define GUI_H_

// GLM headers
#include <glm/glm.hpp>

// Standard headers
#include <memory>

// Engine headers
#include "../backend.hpp"
#include "../buffer_manager.hpp"

namespace mercury {

// Core definitions for GUI
namespace gui {
// Vertex data
struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	// Get vertex binding description
	static VertexBinding vertex_binding() {
		return VertexBinding {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};
	}

	// Get vertex attribute descriptions
	static std::array <VertexAttribute, 2> vertex_attributes() {
		return std::array <VertexAttribute, 2> {
			VertexAttribute {
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, pos)
			},

			VertexAttribute {
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color)
			}
		};
	}
};

// Aliases
using VertexBuffer = BufferManager <Vertex>;
using IndexBuffer = BufferManager <uint32_t>;

// RenderPacket structure contains
// 	the data needed to render all
// 	the GUI elements in a Layer object
struct RenderPacket {
	// Rectangles
	struct {
		VertexBuffer *vb;
		IndexBuffer *ib;
	} rects;

	// TODO: text renders so that text can become an object

	// Reset the render packet
	void reset() {
		rects.vb->reset_push_back();
		rects.ib->reset_push_back();
	}

	// Sync the render packet
	void sync() {
		// Sync sizes
		rects.vb->sync_size();
		rects.ib->sync_size();

		// Upload
		rects.vb->upload();
		rects.ib->upload();
	}
};


// Abstract GUI element type
struct _element {
	// Virtual destructor
	virtual ~_element() {}

	// Child elements
	std::vector <std::shared_ptr <_element>> children;

	// Pure virtual function to render
	virtual void render(RenderPacket &) = 0;

	// Wrapper function to render
	void render_element(RenderPacket &packet) {
		// Render this
		render(packet);

		// Render all children
		for (auto &child : children)
			child->render(packet);
	}
};

// Aliases
using Element = std::shared_ptr <_element>;

}

}

#endif
