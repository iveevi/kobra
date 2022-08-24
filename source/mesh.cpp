// Standard headers
#include <thread>

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
	auto out = Submesh {vertices, indices};
	Mesh m = std::vector <Submesh> {out};
	m._source = "box";
	return m;
}

/*
 * TODO: only parameters should be slices and stacks (transform will do the
 * rest)
kmesh kmesh::make_sphere(const glm::vec3 &center, float radius, int slices, int stacks)
{
	// vertices and indices
	vertexlist vertices;
	indexlist indices;

	// add top vertex
	glm::vec3 top_vertex {center.x, center.y + radius, center.z};
	vertices.push_back(vertex {
		top_vertex,
		{0.0f, 1.0f, 0.0f},
		{0.5f, 0.5f}
	});

	// generate vertices in the middle stacks
	for (int i = 0; i < stacks - 1; i++) {
		float phi = glm::pi <float> () * double(i + 1) / stacks;

		// generate vertices in the slice
		for (int j = 0; j < slices; j++) {
			float theta = 2.0f * glm::pi <float> () * double(j) / slices;

			// add vertex
			// todo: utlility function to generate polar and
			// spherical coordinates
			glm::vec3 vertex {
				center.x + radius * glm::sin(phi) * glm::cos(theta),
				center.y + radius * glm::cos(phi),
				center.z + radius * glm::sin(phi) * glm::sin(theta)
			};

			glm::vec3 normal {
				glm::sin(phi) * glm::cos(theta),
				glm::cos(phi),
				glm::sin(phi) * glm::sin(theta)
			};

			glm::vec2 uv {
				double(j) / slices,
				double(i) / stacks
			};

			// add vertex
			vertices.push_back(vertex {
				vertex,
				normal,
				uv
			});
		}
	}

	// add bottom vertex
	glm::vec3 bottom_vertex {center.x, center.y - radius, center.z};
	vertices.push_back(vertex {
		bottom_vertex,
		{0.0f, -1.0f, 0.0f},
		{0.5f, 0.5f}
	});

	// top and bottom triangles
	for (int i = 0; i < slices; i++) {
		// corresponding top
		int i0 = i + 1;
		int i1 = (i + 1) % slices + 1;

		indices.push_back(0);
		indices.push_back(i1);
		indices.push_back(i0);

		// corresponding bottom
		i0 = i + slices * (stacks - 2) + 1;
		i1 = (i + 1) % slices + slices * (stacks - 2) + 1;

		indices.push_back(vertices.size() - 1);
		indices.push_back(i0);
		indices.push_back(i1);
	}

	// middle triangles
	for (int i = 0; i < stacks - 2; i++) {
		for (int j = 0; j < slices; j++) {
			int i0 = i * slices + j + 1;
			int i1 = i * slices + (j + 1) % slices + 1;
			int i2 = (i + 1) * slices + (j + 1) % slices + 1;
			int i3 = (i + 1) * slices + j + 1;

			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);

			indices.push_back(i0);
			indices.push_back(i2);
			indices.push_back(i3);
		}
	}

	// construct and return the mesh
	return kmesh {vertices, indices};
} */

/* create a ring
kmesh kmesh::make_ring(const glm::vec3 &center, float radius, float width, float height, int slices)
{
	// vertices and indices
	vertexlist vertices;
	indexlist indices;

	// add vertices
	for (int i = 0; i < slices; i++) {
		float theta = 2.0f * glm::pi <float> () * double(i) / slices;
		float iradius = radius - width / 2.0f;
		float oradius = radius + width / 2.0f;

		// upper-inner ring
		vertex u_inner {
			{
				center.x + iradius * glm::sin(theta),
				center.y + height / 2.0f,
				center.z + iradius * glm::cos(theta)
			},
			{0.0f, 1.0f, 0.0f},
			{double(i) / slices, 0.0f}
		};

		// upper-outer ring
		vertex u_outer {
			{
				center.x + oradius * glm::sin(theta),
				center.y + height / 2.0f,
				center.z + oradius * glm::cos(theta)
			},
			{0.0f, 1.0f, 0.0f},
			{double(i) / slices, 1.0f}
		};

		// lower-inner ring
		vertex l_inner {
			{
				center.x + iradius * glm::sin(theta),
				center.y - height / 2.0f,
				center.z + iradius * glm::cos(theta)
			},
			{0.0f, -1.0f, 0.0f},
			{double(i) / slices, 0.0f}
		};

		// lower-outer ring
		vertex l_outer {
			{
				center.x + oradius * glm::sin(theta),
				center.y - height / 2.0f,
				center.z + oradius * glm::cos(theta)
			},
			{0.0f, -1.0f, 0.0f},
			{double(i) / slices, 1.0f}
		};

		// add vertices
		vertices.push_back(u_inner);
		vertices.push_back(u_outer);
		vertices.push_back(l_inner);
		vertices.push_back(l_outer);
	}

	// add indices
	for (int i = 0; i < slices; i++) {
		// next index
		int n = (i + 1) % slices;

		// relevant indices
		uint i0 = 4 * i;
		uint i1 = i0 + 1;
		uint i2 = i0 + 2;
		uint i3 = i0 + 3;

		uint i4 = 4 * n;
		uint i5 = i4 + 1;
		uint i6 = i4 + 2;
		uint i7 = i4 + 3;

		// group indices
		std::vector <uint> upper_quad {
			i0, i4, i5,
			i0, i5, i1
		};

		std::vector <uint> lower_quad {
			i2, i6, i7,
			i2, i7, i3
		};

		std::vector <uint> outer_mid {
			i1, i5, i3,
			i3, i7, i5
		};

		std::vector <uint> inner_mid {
			i0, i4, i2,
			i2, i6, i4
		};

		// add indices
		indices.insert(indices.end(), upper_quad.begin(), upper_quad.end());
		indices.insert(indices.end(), lower_quad.begin(), lower_quad.end());
		indices.insert(indices.end(), outer_mid.begin(), outer_mid.end());
		indices.insert(indices.end(), inner_mid.begin(), inner_mid.end());
	}

	// construct and return the mesh
	return kmesh {vertices, indices};
} */

static Submesh process_mesh(aiMesh *mesh, const aiScene *scene)
{
	// Mesh data
	VertexList vertices;
	Indices indices;

	std::cout << "\tProcessing mesh with " << mesh->mNumVertices
		<< " vertices and " << mesh->mNumFaces << " faces" << std::endl;

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

	// Material
	Material mat;

	aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
	
	// Get diffuse
	aiString path;
	if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
		std::cout << "\t\tFound diffuse texture: " << path.C_Str() << std::endl;
	} else {
		std::cout << "\t\tNo diffuse texture found" << std::endl;
		aiColor3D diffuse;
		material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);

		std::cout << "\t\tDiffuse color: " << diffuse.r << ", " << diffuse.g << ", " << diffuse.b << std::endl;
		mat.diffuse = {diffuse.r, diffuse.g, diffuse.b};
	}

	return Submesh {vertices, indices, mat};
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
	// TODO: multithreaded task system (using a threadpool, etc) in core/
	// std::vector <Mesh> children;

	for (size_t i = 0; i < node->mNumChildren; i++) {
		std::cout << "Processing child " << i << std::endl;
		Mesh m = process_node(node->mChildren[i], scene);

		for (size_t j = 0; j < m.submeshes.size(); j++)
			submeshes.push_back(m.submeshes[j]);
	}

	return Mesh {submeshes};
}

std::optional <Mesh> Mesh::load(const std::string &path)
{
	// Special cases
	if (path == "box")
		return box({0, 0, 0}, {0.5, 0.5, 0.5});

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
			| aiProcess_GenNormals
			| aiProcess_FlipUVs
	);

	// Check if the scene was loaded
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
			|| !scene->mRootNode) {
		Logger::error("[Mesh] Could not load scene: " + path);
		return {};
	}

	// Process the scene (root node)
	Mesh m = process_node(scene->mRootNode, scene);
	m._source = path;

	KOBRA_LOG_FUNC(Log::INFO) << "Loaded mesh with " << m.submeshes.size()
		<< " submeshes (#verts = " << m.vertices() << ", #triangles = "
		<< m.triangles() << "), from " << path << std::endl;

	for (auto &s : m.submeshes) {
		std::cout << "\tSubmesh with " << s.vertices.size()
			<< " vertices and " << s.indices.size() / 3 << " triangles"
			<< std::endl;
	}

	return m;
}

}
