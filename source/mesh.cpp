#include "../include/mesh.hpp"
#include "../include/model.hpp"

namespace kobra {

////////////////////
// Mesh factories //
////////////////////

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
		std::ifstream &file) {
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
		Model model(source);
		mesh = model[source_index];
	}

	// Read material header, then material
	std::getline(file, line);
	if (line != "[MATERIAL]") {
		KOBRA_LOG_FUNC(error) << "Expected material header\n";
		return std::nullopt;
	}

	auto mat = Material::from_file(ctx, command_pool, file);
	if (!mat)
		return std::nullopt;

	mesh.set_material(*mat);

	// Return mesh
	return mesh;
}

}
