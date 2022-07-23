#ifndef KOBRA_MESH_H_
#define KOBRA_MESH_H_

// Standard headers
#include <vector>

// Engine headers
#include "bbox.hpp"
#include "bvh.hpp"
#include "transform.hpp"
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

	// Number of triangles
	int triangles() const {
		return indices.size()/3;
	}

	// Generate a BVH for this submesh
	BVHPtr bvh(const Transform &transform) const {
		// Generate list of bounding boxes
		std::vector <BoundingBox> boxes;

		for (int i = 0; i < indices.size(); i += 3) {
			// Get the triangle
			int a = indices[i];
			int b = indices[i + 1];
			int c = indices[i + 2];

			// Get the vertices
			const auto &v1 = vertices[a];
			const auto &v2 = vertices[b];
			const auto &v3 = vertices[c];

			// Get the transformed vertices
			glm::vec3 p1 = transform * v1.position;
			glm::vec3 p2 = transform * v2.position;
			glm::vec3 p3 = transform * v3.position;

			// Create the bounding box
			glm::vec3 min = glm::min(p1, glm::min(p2, p3));
			glm::vec3 max = glm::max(p1, glm::max(p2, p3));

			// Add the bounding box
			boxes.push_back(BoundingBox {min, max});
		}

		// Generate the BVH
		return partition(boxes);
	}
};

// A mesh is a collection of submeshes
class Mesh {
	// Source file
	std::string _source = "";
public:
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

	// Total number of triangles
	int triangles() const {
		int total = 0;
		for (const auto &submesh : submeshes)
			total += submesh.triangles();
		return total;
	}

	// Get the source file
	const std::string &source() const {
		return _source;
	}

	// Indexing
	const Submesh &operator[](int i) const {
		return submeshes[i];
	}

	// Generate a BVH for this mesh
	BVHPtr bvh(const Transform &transform) const {
		if (submeshes.size() == 1)
			return submeshes[0].bvh(transform);

		// Generate list of partial BVHs
		std::vector <BVHPtr> bvhs;
		for (const auto &submesh : submeshes)
			bvhs.push_back(submesh.bvh(transform));

		// Generate the BVH
		return partition(bvhs);
	}

	// Mesh factories
	// TODO: should make a 1x1x1, then transform will do the rest
	// TODO: clean up and put into source file
	static Mesh box(const glm::vec3 &, const glm::vec3 &);
	static Mesh sphere(const glm::vec3 &, float, int = 16, int = 16);
	static std::optional <Mesh> load(const std::string &);
};

using MeshPtr = std::shared_ptr <Mesh>;

}

#endif
