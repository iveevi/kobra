#ifndef VERTEX_H_
#define VERTEX_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"

namespace kobra {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 tex_coords;

	// Default constructor
	Vertex() = default;

	// Constructors
	Vertex(const glm::vec3 &);
	Vertex(const glm::vec3 &, const glm::vec3 &);
	Vertex(const glm::vec3 &, const glm::vec3 &, const glm::vec2 &);

	// Vertex binding
	static VertexBinding vertex_binding();

	// Get vertex attribute descriptions
	static std::vector <VertexAttribute> vertex_attributes();
};

// Aliases
using VertexList = std::vector <Vertex>;
using IndexList = std::vector <uint32_t>;

}

#endif
