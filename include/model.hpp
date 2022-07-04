#ifndef MODEL_H_
#define MODEL_H_

// Standard headers
#include <fstream>
#include <unordered_map>
#include <vector>

// Assimp headers
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Engine headers
#include "kmesh.hpp"
#include "logger.hpp"

namespace kobra {

// Model structure
// TODO: inherit from object for saving
class Model {
protected:
	// Filename
	std::string		_filename;

	// Meshes
	std::vector <KMesh>	_meshes;

	// Assimp helpers
	void _process_node(aiNode *, const aiScene *);
	void _process_mesh(aiMesh *, const aiScene *);

	// Static cache of previously loaded models
	static std::unordered_map <std::string, Model> _cache;

	// These constructors are private to force
	// 	the use of the factory method
	Model(const char *s);
	Model(const std::string &);
public:
	// Default constructor
	Model() = default;

	// Properties
	size_t mesh_count() const;
	const std::string &filename() const;

	KMesh &operator[](size_t);
	const KMesh &operator[](size_t) const;

	// Load model
	static const Model &load(const std::string &);
};

}

#endif
