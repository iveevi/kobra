#include "../include/mesh.hpp"

namespace kobra {

Mesh Mesh::make_box(const glm::vec3 &center, const glm::vec3 &dim)
{
	float x = dim.x;
	float y = dim.y;
	float z = dim.z;

	// All 24 vertices, with correct normals
	VertexList vertices {
		// Front
		Vertex {{ center.x - x, center.y - y, center.z + z }, { 0.0f, 0.0f, 1.0f }},
		Vertex {{ center.x + x, center.y - y, center.z + z }, { 0.0f, 0.0f, 1.0f }},
		Vertex {{ center.x + x, center.y + y, center.z + z }, { 0.0f, 0.0f, 1.0f }},
		Vertex {{ center.x - x, center.y + y, center.z + z }, { 0.0f, 0.0f, 1.0f }},

		// Back
		Vertex {{ center.x - x, center.y - y, center.z - z }, { 0.0f, 0.0f, -1.0f }},
		Vertex {{ center.x + x, center.y - y, center.z - z }, { 0.0f, 0.0f, -1.0f }},
		Vertex {{ center.x + x, center.y + y, center.z - z }, { 0.0f, 0.0f, -1.0f }},
		Vertex {{ center.x - x, center.y + y, center.z - z }, { 0.0f, 0.0f, -1.0f }},

		// Left
		Vertex {{ center.x - x, center.y - y, center.z + z }, { -1.0f, 0.0f, 0.0f }},
		Vertex {{ center.x - x, center.y - y, center.z - z }, { -1.0f, 0.0f, 0.0f }},
		Vertex {{ center.x - x, center.y + y, center.z - z }, { -1.0f, 0.0f, 0.0f }},
		Vertex {{ center.x - x, center.y + y, center.z + z }, { -1.0f, 0.0f, 0.0f }},

		// Right
		Vertex {{ center.x + x, center.y - y, center.z + z }, { 1.0f, 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z - z }, { 1.0f, 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z - z }, { 1.0f, 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z + z }, { 1.0f, 0.0f, 0.0f }},

		// Top
		Vertex {{ center.x - x, center.y + y, center.z + z }, { 0.0f, 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z + z }, { 0.0f, 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z - z }, { 0.0f, 1.0f, 0.0f }},
		Vertex {{ center.x - x, center.y + y, center.z - z }, { 0.0f, 1.0f, 0.0f }},

		// Bottom
		Vertex {{ center.x - x, center.y - y, center.z + z }, { 0.0f, -1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z + z }, { 0.0f, -1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z - z }, { 0.0f, -1.0f, 0.0f }},
		Vertex {{ center.x - x, center.y - y, center.z - z }, { 0.0f, -1.0f, 0.0f }}
	};

	// All 36 indices
	IndexList indices {
		0, 1, 2,	2, 3, 0,	// Front
		4, 5, 6,	6, 7, 4,	// Back
		8, 9, 10,	10, 11, 8,	// Left
		12, 13, 14,	14, 15, 12,	// Right
		16, 17, 18,	18, 19, 16,	// Top
		20, 21, 22,	22, 23, 20	// Bottom
	};

	return Mesh { vertices, indices };
}

}
