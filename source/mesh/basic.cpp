#include "include/mesh/basic.hpp"

namespace mercury {

namespace mesh {

void add_triangle(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &p1,
		const glm::vec3 &p2,
		const glm::vec3 &p3)
{
	glm::vec3 v1 = p2 - p1;
	glm::vec3 v2 = p3 - p1;
	glm::vec3 normal = glm::cross(v1, v2);

	unsigned int base = vertices.size();

	// Triangle coordinates
	Mesh::AVertex tmp {
		Vertex {.position = p1, .normal = normal},
		Vertex {.position = p2, .normal = normal},
		Vertex {.position = p3, .normal = normal}
	};

	// Add vertices
	vertices.insert(vertices.begin(), tmp.begin(), tmp.end());

	// Add indices
	indices.push_back(base);
	indices.push_back(base + 1);
	indices.push_back(base + 2);
}

void add_face(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &p1,
		const glm::vec3 &p2,
		const glm::vec3 &p3,
		const glm::vec3 &p4,
		const glm::vec3 &normal)
{
	add_triangle(vertices, indices, p1, p3, p2);
	add_triangle(vertices, indices, p1, p4, p3);
}

// Generates a cuboid mesh
Mesh cuboid(const glm::vec3 &center, float w, float h, float d)
{
	float w2 = w/2;
	float h2 = h/2;
	float d2 = d/2;

	// 6 faces
	Mesh::AVertex vertices;
	Mesh::AIndices indices;

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z - d2},
		{center.x + w2, center.y - h2, center.z - d2},
		{center.x + w2, center.y + h2, center.z - d2},
		{center.x - w2, center.y + h2, center.z - d2},
		{0, 0, -1}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z + d2},
		{center.x - w2, center.y + h2, center.z + d2},
		{center.x + w2, center.y + h2, center.z + d2},
		{center.x + w2, center.y - h2, center.z + d2},
		{0, 0, 1}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z - d2},
		{center.x - w2, center.y - h2, center.z + d2},
		{center.x + w2, center.y - h2, center.z + d2},
		{center.x + w2, center.y - h2, center.z - d2},
		{0, -1, 0}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y + h2, center.z - d2},
		{center.x + w2, center.y + h2, center.z - d2},
		{center.x + w2, center.y + h2, center.z + d2},
		{center.x - w2, center.y + h2, center.z + d2},
		{0, 1, 0}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z - d2},
		{center.x - w2, center.y + h2, center.z - d2},
		{center.x - w2, center.y + h2, center.z + d2},
		{center.x - w2, center.y - h2, center.z + d2},
		{-1, 0, 0}
	);

	add_face(vertices, indices,
		{center.x + w2, center.y - h2, center.z - d2},
		{center.x + w2, center.y - h2, center.z + d2},
		{center.x + w2, center.y + h2, center.z + d2},
		{center.x + w2, center.y + h2, center.z - d2},
		{1, 0, 0}
	);

	return Mesh {vertices, {}, indices};
}

}

}
