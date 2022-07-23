// Assimp headers
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Engine headers
#include "../include/mesh.hpp"

namespace kobra {

// Submesh

// Mesh
Mesh Mesh::box(const glm::vec3 &center, const glm::vec3 &dim)
{
	float x = dim.x;
	float y = dim.y;
	float z = dim.z;

	// All 24 vertices, with correct normals
	VertexList vertices {
		// Front
		Vertex {{ center.x - x, center.y - y, center.z + z }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z + z }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z + z }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f }},
		Vertex {{ center.x - x, center.y + y, center.z + z }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f }},

		// Back
		Vertex {{ center.x - x, center.y - y, center.z - z }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z - z }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z - z }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f }},
		Vertex {{ center.x - x, center.y + y, center.z - z }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f }},

		// Left
		Vertex {{ center.x - x, center.y - y, center.z + z }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }},
		Vertex {{ center.x - x, center.y - y, center.z - z }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }},
		Vertex {{ center.x - x, center.y + y, center.z - z }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }},
		Vertex {{ center.x - x, center.y + y, center.z + z }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }},

		// Right
		Vertex {{ center.x + x, center.y - y, center.z + z }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z - z }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z - z }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }},
		Vertex {{ center.x + x, center.y + y, center.z + z }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }},

		// Top
		Vertex {{ center.x - x, center.y + y, center.z + z }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z + z }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y + y, center.z - z }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }},
		Vertex {{ center.x - x, center.y + y, center.z - z }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f }},

		// Bottom
		Vertex {{ center.x - x, center.y - y, center.z + z }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z + z }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }},
		Vertex {{ center.x + x, center.y - y, center.z - z }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f }},
		Vertex {{ center.x - x, center.y - y, center.z - z }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 1.0f }}
	};

	// All 36 indices
	IndexList indices {
		0, 1, 2,	2, 3, 0,	// Front
		4, 6, 5,	6, 4, 7,	// Back
		8, 10, 9,	10, 8, 11,	// Left
		12, 13, 14,	14, 15, 12,	// Right
		16, 17, 18,	18, 19, 16,	// Top
		20, 22, 21,	22, 20, 23	// Bottom
	};

	// TODO: should set source of the mesh to box, then dimensions
	auto out = Submesh { vertices, indices };

	// TODO: source?
	/* out._source = "box";
	out._source_index = 0; */

	return std::vector <Submesh> {out};
}

static Submesh process_mesh(aiMesh *mesh, const aiScene *scene)
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

		// TODO: material?

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

	return Submesh {vertices, indices};
}

static Mesh process_node(aiNode *node, const aiScene *scene)
{
	// Process all the node's meshes (if any)
	std::vector <Submesh> submeshes;
	for (size_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		submeshes.push_back(process_mesh(mesh, scene));
	}

	// Recusively process all the node's children
	for (size_t i = 0; i < node->mNumChildren; i++) {
		Mesh m = process_node(node->mChildren[i], scene);

		for (size_t j = 0; j < m.submeshes.size(); j++)
			submeshes.push_back(m.submeshes[j]);
	}

	return Mesh {submeshes};
}

std::optional <Mesh> Mesh::load(const std::string &path)
{
	// Check if the file exists
	std::ifstream file(path);
	if (!file.is_open()) {
		Logger::error("[Mesh] Could not open file: " + path);
		return {};
	}

	// Create the Assimp importer
	Assimp::Importer importer;

	// Read scene
	const aiScene *scene = importer.ReadFile(
		path, aiProcess_Triangulate
			| aiProcess_GenSmoothNormals
			| aiProcess_FlipUVs
	);

	// Check if the scene was loaded
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
			|| !scene->mRootNode) {
		Logger::error("[Mesh] Could not load scene: " + path);
		return {};
	}

	// Process the scene (root node)
	return process_node(scene->mRootNode, scene);
}

}
