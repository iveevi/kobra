#pragma once

// Standard headers
#include <optional>
#include <vector>

// Engine headers
#include "bbox.hpp"
#include "bvh.hpp"
#include "transform.hpp"
#include "vertex.hpp"
#include "material.hpp"

namespace kobra {

// TODO: inherit mesh from component type?

// Submesh, holds vertices and indices
class Submesh {
	// Process tangent and bitangent
	void _process_vertex_data();
public:
	// Data
	VertexList vertices;
        std::vector <uint32_t> indices;
	uint32_t material_index = 0;

	// Constructors
	// TODO: remove this constructor...
	Submesh(const VertexList &vs, const std::vector <uint32_t> &is,
			uint32_t mat_index = 0,
			bool calculate_tangents = true)
			: vertices(vs), indices(is), material_index(mat_index) {
		/* Process the vertex data
		if (calculate_tangents)
			_process_vertex_data(); */
	}

	// Number of triangles
	int triangles() const {
		return indices.size()/3;
	}

	// Generate a bounding box
	BoundingBox bbox() const;

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

	// Submesh modifiers
	static void transform(Submesh &, const Transform &);

	// Submesh factories
	static Submesh sphere(int = 16, int = 16);
	static Submesh cylinder(int = 32);
	static Submesh cone(int = 32);
};

// A mesh is a collection of submeshes
// TODO: refactor to Model
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

	// Populate mesh cache
	void populate_mesh_cache(std::set <const Submesh *> &cache) {
		for (const auto &submesh : submeshes)
			cache.insert(&submesh);
	}

	// Mesh factories
	// TODO: should make a 1x1x1, then transform will do the rest
	// TODO: clean up and put into source file
	static Mesh box(
		const glm::vec3 & = glm::vec3{0.0f},
		const glm::vec3 & = glm::vec3 {0.5f}
	);

	static Mesh plane(
		const glm::vec3 & = glm::vec3 {0.0f},
		float = 1, float = 1
	);

	static Mesh sphere(const glm::vec3 &, float, int = 16, int = 16);

	static std::optional <Mesh> load(const std::string &);

	// Caching
	static void cache_save(const Mesh &, const std::string &);
	static std::optional <Mesh> cache_load(const std::string &);
};

using MeshPtr = std::shared_ptr <Mesh>;

}
