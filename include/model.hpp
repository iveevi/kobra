#ifndef MODEL_H_
#define MODEL_H_

// Standard headers
#include <vector>
#include <fstream>

// Assimp headers
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Engine headers
#include "mesh.hpp"
#include "logger.hpp"

namespace kobra {

// Model structure
// TODO: inherit from object for saving
class Model {
protected:
	// Filename
	std::string		_filename;
	// Meshes
	std::vector <Mesh>	_meshes;

	// Assimp helpers
	void _process_node(aiNode *, const aiScene *);
	void _process_mesh(aiMesh *, const aiScene *);
public:
	// Constructors
	Model();
	Model(const char *s);
	Model(const std::string &);

	// Properties
	size_t mesh_count() const;

	Mesh &operator[](size_t);
	const Mesh &operator[](size_t) const;
};

}

#endif
