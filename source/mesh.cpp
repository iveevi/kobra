#include "../include/common.hpp"
#include "../include/mesh.hpp"
#include "../include/model.hpp"

namespace kobra {

////////////////////
// Mesh factories //
////////////////////

// Create box
Mesh Mesh::make_box(const glm::vec3 &center, const glm::vec3 &dim)
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
		4, 5, 6,	6, 7, 4,	// Back
		8, 9, 10,	10, 11, 8,	// Left
		12, 13, 14,	14, 15, 12,	// Right
		16, 17, 18,	18, 19, 16,	// Top
		20, 21, 22,	22, 23, 20	// Bottom
	};

	// TODO: should set source of the mesh to box, then dimensions
	return Mesh { vertices, indices };
}

// Create sphere
Mesh Mesh::make_sphere(const glm::vec3 &center, float radius, int slices, int stacks)
{
	// Vertices and indices
	VertexList vertices;
	IndexList indices;

	// Add top vertex
	glm::vec3 top_vertex {center.x, center.y + radius, center.z};
	vertices.push_back(Vertex {
		top_vertex,
		{0.0f, 1.0f, 0.0f},
		{0.5f, 0.5f}
	});

	// Generate vertices in the middle stacks
	for (int i = 0; i < stacks - 1; i++) {
		float phi = glm::pi <float> () * double(i + 1) / stacks;

		// Generate vertices in the slice
		for (int j = 0; j < slices; j++) {
			float theta = 2.0f * glm::pi <float> () * double(j) / slices;

			// Add vertex
			// TODO: utlility function to generate polar and
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

			// Add vertex
			vertices.push_back(Vertex {
				vertex,
				normal,
				uv
			});
		}
	}

	// Add bottom vertex
	glm::vec3 bottom_vertex {center.x, center.y - radius, center.z};
	vertices.push_back(Vertex {
		bottom_vertex,
		{0.0f, -1.0f, 0.0f},
		{0.5f, 0.5f}
	});

	// Top and bottom triangles
	for (int i = 0; i < slices; i++) {
		// Corresponding top
		int i0 = i + 1;
		int i1 = (i + 1) % slices + 1;

		indices.push_back(0);
		indices.push_back(i0);
		indices.push_back(i1);

		// Corresponding bottom
		i0 = i + slices * (stacks - 2) + 1;
		i1 = (i + 1) % slices + slices * (stacks - 2) + 1;

		indices.push_back(vertices.size() - 1);
		indices.push_back(i1);
		indices.push_back(i0);
	}

	// Middle triangles
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

	// Construct and return the mesh
	return Mesh {vertices, indices};
}

//////////////////
// Mesh methods //
//////////////////

void Mesh::save(std::ofstream &file) const
{
	file << "[MESH]" << std::endl;

	// If the mesh was created from a file
	if (_source.size() > 0) {
		file << "source=" << _source << std::endl;
		file << "index=" << _source_index << std::endl;

		// Save material
		_material.save(file);
		return;
	}

	// Otherwise write all vertices and indices
	file << "source=0" << std::endl;

	for (unsigned int i = 0; i < _vertices.size(); i++) {
		glm::vec3 pos = _vertices[i].position;
		glm::vec3 norm = _vertices[i].normal;
		glm::vec2 tex = _vertices[i].tex_coords;

		file << "v " << pos.x << " " << pos.y << " " << pos.z << " "
			<< norm.x << " " << norm.y << " " << norm.z << " "
			<< tex.x << " " << tex.y << std::endl;
	}

	for (unsigned int i = 0; i < _indices.size(); i += 3) {
		file << "f " << _indices[i] + 1 << " "
			<< _indices[i + 1] + 1 << " "
			<< _indices[i + 2] + 1 << std::endl;
	}

	// Save material
	_material.save(file);
}

// Load raw mesh (vertex and index data)
static std::optional <Mesh> load_raw_mesh(std::ifstream &fin)
{
	// Read all vertices
	VertexList vertices;

	std::string line;

	int cpos = fin.tellg();
	while (std::getline(fin, line)) {
		if (line[0] != 'v')
			break;

		cpos = fin.tellg();

		float x, y, z;
		float nx, ny, nz;
		float u, v;

		std::sscanf(line.c_str(),
			"v %f %f %f %f %f %f %f %f",
			&x, &y, &z, &nx, &ny, &nz, &u, &v
		);

		Vertex vtx {
			{ x, y, z },
			{ nx, ny, nz },
			{ u, v }
		};

		vertices.push_back(vtx);
	}

	// Read all indices
	IndexList indices;

	fin.seekg(cpos);
	while (std::getline(fin, line)) {
		if (line[0] != 'f')
			break;

		cpos = fin.tellg();

		int v1, v2, v3;
		std::sscanf(line.c_str(), "f %d %d %d", &v1, &v2, &v3);

		indices.push_back(v1 - 1);
		indices.push_back(v2 - 1);
		indices.push_back(v3 - 1);
	}

	// Go back to the start of the next object
	fin.seekg(cpos);

	// Construct and return mesh
	return Mesh(vertices, indices);
}

// Read from file
std::optional <Mesh> Mesh::from_file(const Vulkan::Context &ctx,
		const VkCommandPool &command_pool,
		std::ifstream &file,
		const std::string &scene_file)
{
	std::string line;

	// Read source
	char buf[1024];
	std::getline(file, line);
	std::sscanf(line.c_str(), "source=%s", buf);
	std::string source = buf;

	// Load mesh
	Mesh mesh;
	if (source == "0") {
		// Load raw mesh
		auto m = load_raw_mesh(file);
		if (!m)
			return std::nullopt;

		mesh = *m;
	} else {
		// Read mesh index
		int source_index = -1;
		std::getline(file, line);
		std::sscanf(line.c_str(), "index=%d", &source_index);

		if (source_index < 0) {
			KOBRA_LOG_FUNC(error) << "Mesh index should not be negative\n";
			return std::nullopt;
		}

		// Load from file
		source = common::get_path(
			source,
			common::get_directory(scene_file)
		);

		const Model &model = Model::load(source);
		mesh = model[source_index];
	}

	// Read material header, then material
	std::getline(file, line);
	if (line != "[MATERIAL]") {
		KOBRA_LOG_FUNC(error) << "Expected material header\n";
		return std::nullopt;
	}

	auto mat = Material::from_file(ctx, command_pool, file, scene_file);
	if (!mat)
		return std::nullopt;

	mesh.set_material(*mat);

	// Return mesh
	return mesh;
}

}
