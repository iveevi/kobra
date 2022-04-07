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
VertexBinding Vertex::vertex_binding()
{
	return VertexBinding {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
	};
}

// Get vertex attribute descriptions
std::vector <VertexAttribute> Vertex::vertex_attributes()
{
	return {
		VertexAttribute {
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position)
		},

		VertexAttribute {
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal)
		},

		VertexAttribute {
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, tex_coords)
		}
	};
}

}
