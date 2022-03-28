// Standard headers
#include <cstdio>
#include <optional>

// Engine headers
#include "../include/scene.hpp"
#include "../include/model.hpp"
#include "../include/mesh.hpp"

namespace kobra {

//////////////////////
// Helper functions //
//////////////////////

static std::optional <Transform> load_transform(std::istream &fin)
{
	// Read the transform header
	std::string header;
	std::getline(fin, header);

	if (header != "[TRANSFORM]") {
		KOBRA_LOG_FUNC(error) << "Expected [TRANSFORM] header, got "
			<< header << std::endl;
		return {};
	}

	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 scale;

	// Read the transform data
	std::getline(fin, header);
	std::sscanf(header.c_str(), "position=%f,%f,%f", &position.x, &position.y, &position.z);

	std::getline(fin, header);
	std::sscanf(header.c_str(), "rotation=%f,%f,%f", &rotation.x, &rotation.y, &rotation.z);

	std::getline(fin, header);
	std::sscanf(header.c_str(), "scale=%f,%f,%f", &scale.x, &scale.y, &scale.z);

	// Create the transform
	return Transform(position, rotation, scale);
}

static ObjectPtr load_raw_mesh(std::ifstream &fin, const Transform &t)
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
			"v %f %f %f %f %f %f",
			&x, &y, &z, &nx, &ny, &nz
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

	// Create the mesh
	Mesh *mesh = new Mesh(vertices, indices, t);
	return ObjectPtr(mesh);
}

static ObjectPtr load_mesh(std::ifstream &fin, const Transform &t)
{
	// Get source
	std::string line;
	
	std::string source;
	std::getline(fin, line);

	// Get substring into source ("source=...")
	// TODO: asser the "source" header
	source = line.substr(line.find("=") + 1);

	if (source == "0")
		return load_raw_mesh(fin, t);

	// Get source index
	int source_index = -1;
	std::getline(fin, line);
	std::sscanf(line.c_str(), "index=%d", &source_index);

	if (source_index < 0) {
		KOBRA_LOG_FUNC(error) << "Invalid source index: " << source_index;
		return nullptr;
	}

	// TODO: cache models loaded
	Model model(source);
	Mesh *mesh = new Mesh(model[source_index], t);
	return ObjectPtr(mesh);
}

static ObjectPtr load_object(std::ifstream &fin)
{
	auto t = load_transform(fin);
	if (!t)
		return nullptr;

	// Read the object header
	std::string header;
	std::getline(fin, header);

	if (header == "<MESH>")
		return load_mesh(fin, t.value());

	// Else
	KOBRA_LOG_FUNC(error) << "Unknown object type: \"" << header << "\"";
	return nullptr;
}

//////////////////
// Constructors //
//////////////////

Scene::Scene(const std::string &filename)
{
	// Open the file
	std::ifstream fin(filename);

	// Check if the file is open
	if (!fin.is_open()) {
		KOBRA_LOG_FUNC(error) << "Could not open file " << filename << "\n";
		return;
	}

	// Load all objects
	while (!fin.eof()) {
		// Break if the rest of the file is empty
		if (fin.peek() == EOF)
			break;

		// Read the next object
		ObjectPtr obj = load_object(fin);

		// Check if the object is valid
		if (obj) {
			// Add the object to the scene
			_objects.push_back(obj);
		} else {
			// Skip the line
			std::string line;
			std::getline(fin, line);

			KOBRA_LOG_FUNC(warn) << "Skipping invalid object\n";
			break;
		}
	}
}

Scene::Scene(const std::vector <Object *> &objs)
{
	for (auto &obj : objs)
		_objects.push_back(ObjectPtr(obj));
}

Scene::Scene(const std::vector <ObjectPtr> &objs)
		: _objects(objs) {}
	
/////////////
// Methods //
/////////////


void Scene::save(const std::string &filename) const
{
	// Open the file
	std::ofstream file(filename);

	// Check if the file is open
	if (!file.is_open()) {
		KOBRA_LOG_FUNC(error) << "Could not open file " << filename << "\n";
		return;
	}

	// Write each object
	for (auto &obj : _objects)
		obj->save_object(file);
}

}
