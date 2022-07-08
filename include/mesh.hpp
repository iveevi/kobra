#ifndef KOBRA_MESH_H_
#define KOBRA_MESH_H_

// Standard headers
#include <vector>

// Engine headers
#include "vertex.hpp"

namespace kobra {

// Submesh, holds vertices and indices
class Submesh {
	// Process tangent and bitangent
	// TODO: source file
	void _process_vertex_data() {
		// Iterate over all vertices
		for (int i = 0; i < vertices.size(); i++) {
			// Get the vertex
			Vertex &v = vertices[i];

			// Calculate tangent and bitangent
			v.tangent = glm::vec3(0.0f);
			v.bitangent = glm::vec3(0.0f);

			// Iterate over all faces
			for (int j = 0; j < indices.size(); j += 3) {
				// Get the face
				int face0 = indices[j];
				int face1 = indices[j + 1];
				int face2 = indices[j + 2];

				// Check if the vertex is part of the face
				if (face0 == i || face1 == i || face2 == i) {
					// Get the face vertices
					const auto &v1 = vertices[face0];
					const auto &v2 = vertices[face1];
					const auto &v3 = vertices[face2];

					glm::vec3 e1 = v2.position - v1.position;
					glm::vec3 e2 = v3.position - v1.position;

					glm::vec2 uv1 = v2.tex_coords - v1.tex_coords;
					glm::vec2 uv2 = v3.tex_coords- v1.tex_coords;

					float r = 1.0f / (uv1.x * uv2.y - uv1.y * uv2.x);
					glm::vec3 tangent = (e1 * uv2.y - e2 * uv1.y) * r;
					glm::vec3 bitangent = (e2 * uv1.x - e1 * uv2.x) * r;

					// Add the tangent and bitangent to the vertex
					v.tangent += tangent;
					v.bitangent += bitangent;
				}
			}

			// Normalize the tangent and bitangent
			v.tangent = glm::normalize(v.tangent);
			v.bitangent = glm::normalize(v.bitangent);
		}
	}
public:
	// Data
	VertexList	vertices;
	Indices		indices;

	// Constructors
	Submesh(const VertexList &vs, const Indices &is, bool calculate_tangents = true)
			: vertices(vs), indices(is) {
		// Process the vertex data
		if (calculate_tangents)
			_process_vertex_data();
	}
};

// A mesh is a collection of submeshes
struct Mesh {
	// Data
	std::vector <Submesh> submeshes;

	// Constructors
	Mesh(const std::vector <Submesh> &sm)
		: submeshes(sm) {}

	// Total number of vertices
	int vertices() const {
		int total = 0;
		for (const auto &submesh : submeshes)
			total += submesh.vertices.size();
		return total;
	}

	// Total number of indices
	int indices() const {
		int total = 0;
		for (const auto &submesh : submeshes)
			total += submesh.indices.size();
		return total;
	}

	// Indexing
	const Submesh &operator[](int i) const {
		return submeshes[i];
	}
};

using MeshPtr = std::shared_ptr <Mesh>;

}

#endif
