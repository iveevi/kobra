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
	int default_scene_index = 0;
	std::string default_save_path = "";
	std::vector <Scene> scenes = {};
	std::vector <std::string> scenes_files = {};

	// Default constructor
	Project() {}

	// Load project from a file
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

	// Load scene
	Scene &load_scene(const Context &context, int index = -1) {
		// If negative, use the default scene
		if (index == -1)
			index = default_scene_index;

		// Skip if the scene is already loaded
		if (scenes[index].ecs)
			return scenes[index];

		// Check if the scene exists
		if (index >= scenes_files.size())
			throw std::runtime_error("Scene does not exist");

		// Load the scene into the scene list
		scenes[index].load(context, scenes_files[index]);
		scenes[index].name = "scene-" + std::to_string(index); // TODO: change this
		return scenes[index];
	}

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

                int name_length = material.name.size();
                stream.write((char *) &name_length, sizeof(int));
                stream.write((char *) &material.name, sizeof(std::string));
                stream.write((char *) &material.diffuse, sizeof(glm::vec3));
                stream.write((char *) &material.specular, sizeof(glm::vec3));
                stream.write((char *) &material.ambient, sizeof(glm::vec3));
                stream.write((char *) &material.emission, sizeof(glm::vec3));

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

		// Collect all SUBMESHES to populate the cache
		// TODO: need to find similar enough meshes (e.g. translated or
		// scaled)
		std::set <const Submesh *> submesh_cache;
		for (auto &scene : scenes)
			scene.populate_mesh_cache(submesh_cache);

		std::cout << "Submesh cache size: " << submesh_cache.size() << std::endl;
		for (auto &submesh : submesh_cache) {
			std::cout << submesh << std::endl;
			std::cout << "Size of transcribed submesh: " << transcribe_submesh(*submesh).size() << std::endl;
		}

		// ID each submesh in the cache
		std::vector <std::pair <const Submesh *, std::string>> submesh_ids;
		std::map <const Submesh *, size_t> submesh_id_map;

		// TODO: use the name of the entity for the mesh...
		size_t id = 0;
		for (auto &submesh : submesh_cache) {
			submesh_ids.push_back({submesh, "submesh-" + std::to_string(id++)});
			submesh_id_map[submesh] = id;
		}

		tf::Taskflow taskflow;
		tf::Executor executor;

		taskflow.for_each(submesh_ids.begin(), submesh_ids.end(),
			[&](const auto &pr) {
				std::cout << "Saving submesh " << pr.first<< std::endl;

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
                        mat.name = "material-" + std::to_string(index);
                }

                // For now, store in the cache directory
                taskflow.for_each(Material::all.begin(), Material::all.end(),
                        [&](const Material &mat) {
				std::cout << "Saving material: " << mat.name << std::endl;

				// TODO: ID each submesh in the cache...
				std::filesystem::path filename = cache_path/(mat.name + ".mat");
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

			// Write the scene description
			for (auto &entity : *scene.ecs) {
				file << "\n@entity " << entity.name << "\n";

				const Transform &transform = entity.get <Transform> ();
				file << ".transform "
					<< transform.position.x << " " << transform.position.y << " " << transform.position.z << " "
					<< transform.rotation.x << " " << transform.rotation.y << " " << transform.rotation.z << " "
					<< transform.scale.x << " " << transform.scale.y << " " << transform.scale.z << "\n";

				if (entity.exists <Mesh> ()) {
					file << ".mesh " << entity.get <Mesh> ().submeshes.size();
					auto &submeshes = entity.get <Mesh> ().submeshes;
					for (auto &submesh : submeshes) {
						size_t id = submesh_id_map[&submesh];
						file << " submesh-" << id << ", " << submesh.material_index;
					}
					file << "\n";

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
			file << scene.name << "\n";

		// TODO: use fmt library

		// TODO: other project information

		file.close();
	}
};

}
