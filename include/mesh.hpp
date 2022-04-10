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

	// Vertices and indices
	VertexList 	_vertices;
	Indices		_indices;

	// Process tangent and bitangent
	void _process_vertex_data() {
		// Iterate over all vertices
		for (int i = 0; i < _vertices.size(); i++) {
			// Get the vertex
			Vertex &v = _vertices[i];

			// Calculate tangent and bitangent
			v.tangent = glm::vec3(0.0f);
			v.bitangent = glm::vec3(0.0f);

			// Iterate over all faces
			for (int j = 0; j < _indices.size(); j += 3) {
				// Get the face
				int face0 = _indices[j];
				int face1 = _indices[j + 1];
				int face2 = _indices[j + 2];

				// Check if the vertex is part of the face
				if (face0 == i || face1 == i || face2 == i) {
					// Get the face vertices
					const auto &v1 = _vertices[face0];
					const auto &v2 = _vertices[face1];
					const auto &v3 = _vertices[face2];

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
	// Default constructor
	Mesh() = default;

	// Simple constructor
	// TODO: load tangent and bitangent in loading models
	Mesh(const Mesh &mesh, const Transform &transform)
			: Object(object_type, transform),
			Renderable(mesh.material()),
			_source(mesh._source),
			_source_index(mesh._source_index),
			_vertices(mesh._vertices),
			_indices(mesh._indices) {}

	Mesh(const VertexList &vs, const Indices &is,
			const Transform &t = Transform(),
			bool calculate_tangents = true)
			: Object(object_type, t),
			_vertices(vs), _indices(is) {
		// Process the vertex data
		if (calculate_tangents)
			_process_vertex_data();
	}

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
