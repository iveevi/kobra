#ifndef MESH_H_
#define MESH_H_

// Engine headers
#include "primitive.hpp"
#include "logger.hpp"	// TODO: remove
#include "transform.hpp"
#include "vertex.hpp"
#include "world.hpp"
#include "world_update.hpp"

namespace kobra {

// Mesh class, just holds a list of vertices and indices
template <VertexType T = VERTEX_TYPE_POSITION>
class Mesh {
protected:
	// List of vertices
	VertexList <T>	_vertices;

	// List of indices
	Indices		_indices;
public:
	// Simple constructor
	Mesh() {}
	Mesh(const VertexList <T> &vs, const Indices &is)
			: _vertices {vs}, _indices {is} {}

	// Properties
	size_t vertex_count() const {
		return _vertices.size();
	}

	size_t triangle_count() const {
		return _indices.size() / 3;
	}

	// Get data
	const VertexList <T> &vertices() const {
		return _vertices;
	}

	const Indices &indices() const {
		return _indices;
	}

	/* Count primitives
	uint count() const override {
		return triangle_count();
	} */

	// TODO: override save method
};

// Mesh for raytracing
namespace raytracing {

// Mesh class
template <VertexType T = VERTEX_TYPE_POSITION>
class Mesh : public Primitive, public kobra::Mesh <T> {
public:
	//
	Mesh() {}

	// TODO: are these obselete?
	Mesh(const VertexList <T> &vertices, const Indices &indices,
			const Material &material)
			: Primitive {OBJECT_TYPE_NONE, Transform(), material},
			kobra::Mesh <T> {vertices, indices} {}

	Mesh(const VertexList <T> &vertices, const Indices &indices,
			const Transform &trans, const Material &mat)
			: Primitive {OBJECT_TYPE_NONE, trans, mat},
			kobra::Mesh <T> {vertices, indices} {}

	// Virtual methods
	uint count() const override {
		return this->triangle_count();
	}

	// Write to file
	void save(std::ofstream &file) const override {
		// Header for object
		file << "Mesh\n";

		// Write vertices in binary
		file << "\tvertices:";
		file.write(reinterpret_cast <const char *> (
			&this->_vertices[0]),
			sizeof(Vertex <T>) * this->_vertices.size()
		);

		file << "\n";

		// Write indices in binary
		file << "\tindices:";
		file.write(reinterpret_cast <const char *> (
			&this->_indices[0]),
			sizeof(uint32_t) * this->_indices.size()
		);

		file << "\n";
	}

	// Write mesh to buffer (fake to resolve abstract base class)
	void write(WorldUpdate &) const override {
		// Throw
		throw std::runtime_error("Mesh::write not implemented");
	}

	// Write mesh to buffer
	// TODO: write to both vertex and object buffers
	void write_object(WorldUpdate &wu) override {
		// Remove last index
		wu.indices.erase(wu.indices.end() - 1);

		// Get index of material and push
		uint mati = wu.bf_mats->push_size();
		material.write_material(wu);

		// Get index of transform and push
		uint tati = wu.bf_trans->push_size();
		wu.bf_trans->push_back(transform.model());

		// Push all vertices
		uint offset = wu.bf_verts->push_size();
		for (const auto &v : this->_vertices)
			wu.bf_verts->push_back({v.pos, 1.0});

		// Dummy triangle instance
		Triangle triangle {
			glm::vec3 {0.0f, 0.0f, 0.0f},
			glm::vec3 {0.0f, 0.0f, 0.0f},
			glm::vec3 {0.0f, 0.0f, 0.0f},
			material
		};

		// Write indices
		for (size_t i = 0; i < this->_indices.size(); i += 3) {
			wu.indices.push_back(wu.bf_objs->push_size());
			triangle.write_indexed(wu,
				this->_indices[i] + offset,
				this->_indices[i + 1] + offset,
				this->_indices[i + 2] + offset,
				mati,
				tati
			);
		}
	}

	// Get bounding boxes
	void extract_bboxes(std::vector <kobra::BoundingBox> &bboxes, const glm::mat4 &parent) const override {
		// Get combined transform
		glm::mat4 combined = parent * transform.model();

		// Get bounding box for each triangle
		for (size_t i = 0; i < this->_indices.size(); i += 3) {
			// Get each vertex
			const Vertex <T> &v0 = this->_vertices[this->_indices[i + 0]];
			const Vertex <T> &v1 = this->_vertices[this->_indices[i + 1]];
			const Vertex <T> &v2 = this->_vertices[this->_indices[i + 2]];

			// Construct triangle
			Triangle triangle {
				v0.pos, v1.pos, v2.pos
			};

			// Add bounding box
			triangle.extract_bboxes(bboxes, combined);
		}
	}
};

}

}

#endif
