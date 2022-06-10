#include "../include/vertex.hpp"
#include <vulkan/vulkan_structs.hpp>

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
vk::VertexInputBindingDescription Vertex::vertex_binding()
{
	return {
		0, sizeof(Vertex),
		vk::VertexInputRate::eVertex
	};
}

// Get vertex attribute descriptions
std::vector <vk::VertexInputAttributeDescription> Vertex::vertex_attributes()
{
	return {
		vk::VertexInputAttributeDescription {
			0, 0,
			vk::Format::eR32G32B32Sfloat,
			offsetof(Vertex, position)
		},
		vk::VertexInputAttributeDescription {
			1, 0,
			vk::Format::eR32G32B32Sfloat,
			offsetof(Vertex, normal)
		},
		
		vk::VertexInputAttributeDescription {
			2, 0,
			vk::Format::eR32G32Sfloat,
			offsetof(Vertex, tex_coords)
		},

		vk::VertexInputAttributeDescription {
			3, 0,
			vk::Format::eR32G32B32Sfloat,
			offsetof(Vertex, tangent)
		},

		vk::VertexInputAttributeDescription {
			4, 0,
			vk::Format::eR32G32B32Sfloat,
			offsetof(Vertex, bitangent)
		}
	};
}

}
