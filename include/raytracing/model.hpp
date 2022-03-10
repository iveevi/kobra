#ifndef RAYTRACING_MODEL_H_
#define RAYTRACING_MODEL_H_

// Standard headers
#include <vector>
#include <fstream>

// Assimp headers
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Engine headers
#include "../model.hpp"
#include "../world_update.hpp" // TODO: move to this directory

namespace kobra {

namespace raytracing {

// Model structure
template <VertexType T>
class Model : public Primitive, public kobra::Model <T> {
public:
	// Constructors
	Model();
	Model(const char *s);
	Model(const std::string &);

	// Properties
	size_t mesh_count() const;

	Mesh <T> &operator[](size_t);
	const Mesh <T> &operator[](size_t) const;

	// Count primitives
	uint count() const override {
		uint c = 0;
		for (auto &m : this->_meshes)
			c += m.count();
		return c;
	}

	// Write to file
	void save(std::ofstream &file) const override;

	// Write model to buffer (fake to resolve abstract base class)
	void write(WorldUpdate &) const override {
		// Throw
		throw std::runtime_error("Model::write not implemented");
	}

	// Write model to buffer
	void write_object(WorldUpdate &wu) override {
		// Write meshes
		for (size_t i = 0; i < this->_meshes.size(); i++) {
			// Write mesh
			this->_meshes[i].write_object(wu);
		}
	}

	// Get bounding boxes
	void extract_bboxes(std::vector <kobra::BoundingBox> &bboxes, const glm::mat4 &parent) const override {
		// Get combined transform
		glm::mat4 combined = parent * transform.model();

		// Get bounding box for each mesh
		for (size_t i = 0; i < this->_meshes.size(); i++) {
			// Get bounding box
			this->_meshes[i].extract_bboxes(bboxes, combined);
		}
	}
};
// Write to file
template <VertexType T>
void Model <T> ::save(std::ofstream &file) const
{
	// Header and write number of meshes
	file << "MESH " << this->_filename << std::endl;
}

}

}

#endif
