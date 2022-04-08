#ifndef MESH_H_
#define MESH_H_

// Standard headers
#include <optional>

// Engine headers
#include "primitive.hpp"
#include "logger.hpp"	// TODO: remove
#include "transform.hpp"
#include "types.hpp"
#include "vertex.hpp"
#include "world_update.hpp"
#include "object.hpp"
#include "renderable.hpp"

namespace kobra {

// Mesh class, just holds a list of vertices and indices
class Mesh : virtual public Object, virtual public Renderable {
public:
	static constexpr char object_type[] = "Mesh";
protected:
	// Potential source
	std::string	_source;
	int		_source_index = -1;

	// List of vertices
	VertexList 	_vertices;

	// List of indices
	Indices		_indices;
public:
	// Default constructor
	Mesh() = default;

	// Simple constructor
	Mesh(const Mesh &mesh, const Transform &transform)
			: Object(object_type, transform),
			Renderable(mesh.material()),
			_source(mesh._source),
			_source_index(mesh._source_index),
			_vertices(mesh._vertices),
			_indices(mesh._indices) {}

	Mesh(const VertexList &vs, const Indices &is,
			const Transform &t = Transform())
			: Object(object_type, t),
			_vertices(vs), _indices(is) {}

	// Properties
	size_t vertex_count() const {
		return _vertices.size();
	}

	size_t triangle_count() const {
		return _indices.size() / 3;
	}

	// Centroid of the mesh
	glm::vec3 centroid() const {
		glm::vec3 s {0.0f, 0.0f, 0.0f};

		// Sum the centroids of all triangles
		for (size_t i = 0; i < _indices.size(); i += 3) {
			s += _vertices[_indices[i]].position;
			s += _vertices[_indices[i + 1]].position;
			s += _vertices[_indices[i + 2]].position;
		}

		// Divide by the number of triangles
		return s / (3.0f * triangle_count());
	}

	// Get data
	const VertexList &vertices() const {
		return _vertices;
	}

	const Indices &indices() const {
		return _indices;
	}

	// Virtual methods
	void save(std::ofstream &) const override;

	// Mesh factories
	static Mesh make_box(const glm::vec3 &, const glm::vec3 &);
	static Mesh make_sphere(const glm::vec3 &, float, int = 16, int = 16);

	// Read from file
	static std::optional <Mesh> from_file(const Vulkan::Context &,
		const VkCommandPool &, std::ifstream &, const std::string &);

	// Friends
	friend class Model;
};

}

#endif
