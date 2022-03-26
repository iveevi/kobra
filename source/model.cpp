#include "../include/model.hpp"

namespace kobra {

// Assimp helpers
void Model::_process_node(aiNode *node, const aiScene *scene)
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

void Model::_process_mesh(aiMesh *mesh, const aiScene *scene)
{
	// Mesh data
	VertexList vertices;
	Indices indices;

	// Process all the mesh's vertices
	for (size_t i = 0; i < mesh->mNumVertices; i++) {
		// Create a new vertex
		Vertex v;

		// Vertex position
		v.position = {
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		};

		// Vertex normal
		if (mesh->HasNormals()) {
			v.normal = {
				mesh->mNormals[i].x,
				mesh->mNormals[i].y,
				mesh->mNormals[i].z
			};
		}

		// Vertex texture coordinates
		if (mesh->HasTextureCoords(0)) {
			v.tex_coords = {
				mesh->mTextureCoords[0][i].x,
				mesh->mTextureCoords[0][i].y
			};
		}

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
	_meshes.push_back(Mesh (vertices, indices));
}

// Constructors
Model::Model() {}
Model::Model(const char *path) : Model(std::string(path)) {}

Model::Model(const std::string &filename) : _filename(filename)
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
size_t Model::mesh_count() const
{
	return _meshes.size();
}

Mesh &Model::operator[](size_t i)
{
	return _meshes[i];
}

const Mesh &Model::operator[](size_t i) const
{
	return _meshes[i];
}

}
