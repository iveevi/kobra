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
};

// Aliases
using VertexBuffer = BufferManager <Vertex>;
using IndexBuffer = BufferManager <uint32_t>;

// Abstract object template
struct Object {
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
using ObjectPtr = std::shared_ptr <Object>;

}

}

#endif