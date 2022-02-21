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
#include "world_update.hpp"

namespace mercury {

// Model structure
template <VertexType T>
class Model : public Primitive {
	// Filename
	std::string		_filename;
	// Meshes
	std::vector <Mesh <T>>	_meshes;
	
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

	Mesh <T> &operator[](size_t);
	const Mesh <T> &operator[](size_t) const;

	// Count primitives
	uint count() const override {
		uint c = 0;
		for (auto &m : _meshes)
			c += m.count();
		return c;
	}

	// Write to file
	void save(std::ofstream &file) const override;

	// Write model to buffer (fake to resolve abstract base class)
	void write(Buffer &buffer) const override {
		// Throw
		throw std::runtime_error("Model::write not implemented");
	}

	// Write model to buffer
	void write_object(WorldUpdate &wu) override {
		// Write meshes
		for (size_t i = 0; i < _meshes.size(); i++) {
			// Write mesh
			_meshes[i].write_object(wu);
		}
	}

	// Get bounding boxes
	void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const override {
		// Get bounding box for each mesh
		for (size_t i = 0; i < _meshes.size(); i++) {
			// Get bounding box
			_meshes[i].extract_bboxes(bboxes);
		}
	}
};

// Assimp helpers
template <VertexType T>
void Model <T> ::_process_node(aiNode *node, const aiScene *scene)
{
	// Process all the node's meshes (if any)
	for (size_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		_process_mesh(mesh, scene);
	}

	// Recusively process all the node's children
	for (size_t i = 0; i < node->mNumChildren; i++)
		_process_node(node->mChildren[i], scene);
}

template <VertexType T>
void Model <T> ::_process_mesh(aiMesh *mesh, const aiScene *scene)
{
	// Mesh data
	typename Mesh <T> ::VertexList vertices;
	Indices indices;

	// Process all the mesh's vertices
	for (size_t i = 0; i < mesh->mNumVertices; i++) {
		// Create a new vertex
		Vertex <T> v;

		// Only consider position
		v.pos = {
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		};

		vertices.push_back(v);
	}

	// Process all the mesh's indices
	for (size_t i = 0; i < mesh->mNumFaces; i++) {
		// Get the face
		aiFace face = mesh->mFaces[i];

		// Process all the face's indices
		for (size_t j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}

	// TODO: ignoring materials right now
	
	// Push back the mesh, and its material
	// 	default material is magenta
	Material mat {
		.albedo = glm::vec3 {
			1.0f, 0.0f, 1.0f
		},
		.shading = SHADING_TYPE_FLAT
	};

	_meshes.push_back(Mesh <T> (vertices, indices, mat));
}

// Constructors
template <VertexType T>
Model <T>::Model() {}

template <VertexType T>
Model <T> ::Model(const char *path) : Model(std::string(path)) {}

template <VertexType T>
Model <T> ::Model(const std::string &filename) : _filename(filename)
{
	// Check if the file exists
	std::ifstream file(filename);
	if (!file.is_open()) {
		Logger::error("[Mesh] Could not open file: " + filename);
		return;
	}

	// Create the Assimp importer
	Assimp::Importer importer;

	// Read scene
	const aiScene *scene = importer.ReadFile(
		filename, aiProcess_Triangulate
			| aiProcess_GenSmoothNormals
			| aiProcess_FlipUVs
	);

	// Check if the scene was loaded
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
			|| !scene->mRootNode) {
		Logger::error("[Mesh] Could not load scene: " + filename);
		return;
	}

	// Process the scene (root node)
	_process_node(scene->mRootNode, scene);
}

// Properties
template <VertexType T>
size_t Model <T> ::mesh_count() const
{
	return _meshes.size();
}

template <VertexType T>
Mesh <T> &Model <T> ::operator[](size_t i)
{
	return _meshes[i];
}

template <VertexType T>
const Mesh <T> &Model <T> ::operator[](size_t i) const
{
	return _meshes[i];
}

// Write to file
template <VertexType T>
void Model <T> ::save(std::ofstream &file) const
{
	// Header and write number of meshes
	file << "MESH " << _filename << std::endl;
}

}

#endif
