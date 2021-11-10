#ifndef BASIC_H_
#define BASIC_H_

// Engine headers
#include "include/model.hpp"

namespace mercury {

namespace mesh {

// Add triangle vertices
void add_triangle(Mesh::AVertex &, Mesh::AIndices &,
		const glm::vec3 &, const glm::vec3 &, const glm::vec3 &);

// Generates a cuboid mesh
Mesh cuboid(const glm::vec3 &, float, float, float);

}

}

#endif
