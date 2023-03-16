#pragma once

// Standard headers
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
		return scenes[index];
	}

	// Save project
	void save(const std::string &dir) {
		printf("Saving to %s\n", dir.c_str());
		std::filesystem::path p = dir;

		// Create the necessary directories
		std::filesystem::create_directory(p);			// Root directory
		std::filesystem::create_directory(p / ".cache");	// Cache directory
		std::filesystem::create_directory(p / "assets");	// Assets directory (user home)

		// Collect all SUBMESHES to populate the cache
		// TODO: need to find similar enough meshes (e.g. translated or
		// scaled)
		std::set <MeshPtr> mesh_cache;
		for (auto &scene : scenes)
			scene.populate_mesh_cache(mesh_cache);

		std::cout << "Mesh cache size: " << mesh_cache.size() << std::endl;
		for (auto &mesh : mesh_cache)
			std::cout << mesh.get() << std::endl;
	}
};

}
