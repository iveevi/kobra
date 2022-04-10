#include "../include/vertex.hpp"

namespace kobra {

//////////////////
// Constructors //
//////////////////

Vertex::Vertex(const glm::vec3 &pos)
		: position(pos) {}

Vertex::Vertex(const glm::vec3 &pos, const glm::vec3 &n)
		: position(pos), normal(n) {}

Vertex::Vertex(const glm::vec3 &pos, const glm::vec3 &n, const glm::vec2 &tc)
		: position(pos), normal(n), tex_coords(tc) {}

////////////////////
// Static methods //
////////////////////

// Vertex binding
Vulkan::VB Vertex::vertex_binding()
{
	return Vulkan::VB {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
	};
}

// Get vertex attribute descriptions
std::vector <Vulkan::VA> Vertex::vertex_attributes()
{
	return {
		Vulkan::VA {
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position)
		},

		Vulkan::VA {
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal)
		},

		Vulkan::VA {
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, tex_coords)
		},

		Vulkan::VA {
			.location = 3,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, tangent)
		},

		Vulkan::VA {
			.location = 4,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, bitangent)
		}
	};
}

}
