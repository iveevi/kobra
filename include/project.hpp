#pragma once

// Standard headers
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Taskflow headers
#include <taskflow/taskflow.hpp>

// Engine headers
#include "scene.hpp"

namespace kobra {

// Project description; for a complete game, rendering application, etc.
struct Project {
	std::string directory = "";

	int default_scene_index = 0;
	std::string default_save_path = "";
	std::vector <Scene> scenes = {};
	std::vector <std::string> scenes_files = {};

	// Default constructor
	Project() {}

	// Load project from a file
	// NOTE: currently a bootstrap
	Project(const std::string &project_file) {
		// Check if the file exists
		if (!std::filesystem::exists(project_file)) // TODO: kobra error...
			throw std::runtime_error("Project file does not exist");

		std::ifstream file(project_file);

		std::string scene_file;
		while (file >> scene_file)
			scenes_files.push_back(scene_file);

		// Resize (but all are null)
		scenes.resize(scenes_files.size());

		// For now, just load the first scene
		default_scene_index = 0;
	}

	// Load project from directory
	void load_project(const std::string &dir) {
		directory = dir;

		printf("Loading project from path: %s\n", dir.c_str());
		std::filesystem::path path = dir;

		// Project file
		std::filesystem::path project_file = path / "project.den";
		if (!std::filesystem::exists(project_file))
			throw std::runtime_error("Project file does not exist");

		// Load the project file
		std::string token;

		std::ifstream file(project_file);

		int number_of_scenes;
		file >> token >> number_of_scenes;
		if (token != "@scenes")
			throw std::runtime_error("Invalid project file");

		// Load the scenes
		printf("Loading %d scenes\n", number_of_scenes);
		for (int i = 0; i < number_of_scenes; i++) {
			std::string scene_file;
			file >> scene_file;
			scenes_files.push_back(scene_file);

			printf("Scene %d: %s\n", i, scene_file.c_str());

			// Ensure the scene file exists
			if (!std::filesystem::exists(path / scene_file))
				throw std::runtime_error("Scene file does not exist");
		}

		// Resize (but all are null)
		scenes.resize(scenes_files.size());
	}

	// Load scene
	Scene &load_scene(const Context &, int = -1);

	// Transcribe submesh into binary data
	static std::string transcribe_submesh(const Submesh &submesh) {
		// TODO: generate the tangent and bitangent vectors
		// if they are not present yet

		// Setup the stream
		std::ostringstream stream;

		int vertices = submesh.vertices.size();
		int indices = submesh.indices.size();

		stream.write((char *) &vertices, sizeof(int));
		stream.write((char *) &indices, sizeof(int));
		stream.write((char *) submesh.vertices.data(), sizeof(Vertex) * vertices);
		stream.write((char *) submesh.indices.data(), sizeof(int) * indices);

		return stream.str();
	}

        // Transcribe material into binary data
        static std::string transcribe_material(const Material &material) {
                // Setup the stream
                std::ostringstream stream;

		printf("Transcribing material %s\n", material.name.c_str());
                int name_length = material.name.size();
                stream.write((char *) &name_length, sizeof(int));
                stream.write(material.name.c_str(), name_length * sizeof(char));
                stream.write((char *) &material.diffuse, sizeof(glm::vec3));
                stream.write((char *) &material.specular, sizeof(glm::vec3));
                stream.write((char *) &material.ambient, sizeof(glm::vec3));
                stream.write((char *) &material.emission, sizeof(glm::vec3));
                stream.write((char *) &material.roughness, sizeof(float));
                stream.write((char *) &material.refraction, sizeof(float));
		stream.write((char *) &material.type, sizeof(Shading));

		int normal_texture_length = material.normal_texture.size();
		stream.write((char *) &normal_texture_length, sizeof(int));
		stream.write(material.normal_texture.c_str(), normal_texture_length * sizeof(char));

		int diffuse_texture_length = material.albedo_texture.size();
		stream.write((char *) &diffuse_texture_length, sizeof(int));
		stream.write(material.albedo_texture.c_str(), diffuse_texture_length * sizeof(char));

		int specular_texture_length = material.specular_texture.size();
		stream.write((char *) &specular_texture_length, sizeof(int));
		stream.write(material.specular_texture.c_str(), specular_texture_length * sizeof(char));

		int emission_texture_length = material.emission_texture.size();
		stream.write((char *) &emission_texture_length, sizeof(int));
		stream.write(material.emission_texture.c_str(), emission_texture_length * sizeof(char));

		int roughness_texture_length = material.roughness_texture.size();
		stream.write((char *) &roughness_texture_length, sizeof(int));
		stream.write(material.roughness_texture.c_str(), roughness_texture_length * sizeof(char));

                return stream.str();
        }

	// Save project
	void save(const std::string &dir) {
		// TODO: detect parts that have changed...
		printf("Saving to %s\n", dir.c_str());
		std::filesystem::path path = dir;

		// Create the necessary directories
		std::filesystem::create_directory(path);			// Root directory
		std::filesystem::create_directory(path / ".cache");	// Cache directory
		std::filesystem::create_directory(path / "assets");	// Assets directory (user home)

		std::filesystem::path cache_path = path / ".cache";
		std::filesystem::path assets_path = path / "assets";

		// Collect all SUBMESHES to populate the cache
		// TODO: need to find similar enough meshes (e.g. translated or
		// scaled)
		std::set <const Submesh *> submesh_cache;
		for (auto &scene : scenes)
			scene.populate_mesh_cache(submesh_cache);

		// ID each submesh in the cache
		std::vector <std::pair <const Submesh *, std::string>> submesh_ids;
		std::map <const Submesh *, size_t> submesh_id_map;

		// TODO: use the name of the entity for the mesh...
		size_t id = 0;
		for (auto &submesh : submesh_cache) {
			submesh_ids.push_back({submesh, "submesh-" + std::to_string(id)});
			submesh_id_map[submesh] = id;
			id++;
		}

		tf::Taskflow taskflow;
		tf::Executor executor;

		taskflow.for_each(submesh_ids.begin(), submesh_ids.end(),
			[&](const auto &pr) {
				// TODO: ID each submesh in the cache...
				std::filesystem::path filename = cache_path/(pr.second + ".submesh");
				std::ofstream file(filename, std::ios::binary);

				// Write the submesh to the file
				std::string data = transcribe_submesh(*pr.first);
				file.write(data.data(), data.size());

				file.close();
			}
		);

		executor.run(taskflow).wait();

                // Collect all materials
                // TODO: save in the same location as creation (in the assets
                // directory...)

                int index = 0;
                for (auto &mat : Material::all) {
                        mat.name = "material-" + std::to_string(index++);
                }

                // For now, store in the cache directory
                taskflow.for_each(Material::all.begin(), Material::all.end(),
                        [&](const Material &mat) {
				// TODO: ID each submesh in the cache...
				std::filesystem::path filename = assets_path/(mat.name + ".mat");
				std::ofstream file(filename, std::ios::binary);

				// Write the submesh to the file
				std::string data = transcribe_material(mat);
				file.write(data.data(), data.size());

				file.close();
                        }
                );
		
                executor.run(taskflow).wait();

		// Scene description file (.kobra)
		for (auto &scene : scenes) {
			std::filesystem::path filename = path / (scene.name + ".kobra");
			std::ofstream file(filename);

                        // Write all used materials, in order
                        file << "@materials\n"
				<< ".list " << Material::all.size() << "\n";
                        for (int i = 0; i < Material::all.size(); i++)
                                file << "\tmaterial-" << std::to_string(i) << ".mat\n";

			// Write the scene description
			for (auto &entity : *scene.ecs) {
				file << "\n@entity " << entity.name << "\n";

				const Transform &transform = entity.get <Transform> ();
				file << ".transform "
					<< transform.position.x << " " << transform.position.y << " " << transform.position.z << " "
					<< transform.rotation.x << " " << transform.rotation.y << " " << transform.rotation.z << " "
					<< transform.scale.x << " " << transform.scale.y << " " << transform.scale.z << "\n";

				if (entity.exists <Mesh> ()) {
					file << ".mesh " << entity.get <Mesh> ().submeshes.size() << "\n";
					auto &submeshes = entity.get <Mesh> ().submeshes;
					for (auto &submesh : submeshes) {
						size_t id = submesh_id_map[&submesh];
						file << "\tsubmesh-" << id << ".submesh, " << submesh.material_index << "\n";
					}

					// TODO: material indices...
				}

				if (entity.exists <Renderable> ()) {
					file << ".renderable\n";
					// TODO: material ids
				}

				if (entity.exists <Camera> ()) {
					const Camera &camera = entity.get <Camera> ();
					file << ".camera "
						<< camera.fov << " "
						<< camera.aspect << "\n";
				}
			}

			file.close();
		}

		// Top level project file, describing all the scenes (.den)
		std::filesystem::path filename = path / "project.den";
		std::ofstream file(filename);

		// Write all the scenes
		file << "@scenes " << scenes.size() << "\n";
		for (auto &scene : scenes)
			file << scene.name << ".kobra\n";

		// TODO: use fmt library

		// TODO: other project information

		file.close();
	}
};

}
