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
#include "include/daemons/material.hpp"

namespace kobra {

// Project description; for a complete game, rendering application, etc.
struct Project {
	std::string directory = "";

	int default_scene_index = 0;
	std::string default_save_path = "";
	std::vector <Scene> scenes = {};
	std::vector <std::string> scenes_files = {};

        daemons::MaterialDaemon *material_daemon = nullptr;

	// Default constructor
	Project() = default;

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

                // Allocate material daemon
                material_daemon = daemons::make_material_daemon();
	}

	// Load scene
	Scene &load_scene(const Context &, int = -1);

	// Save project
	void save(const std::string &);

        // Default project
        static Project basic(const Context &context, const std::filesystem::path &dir) {
                Project project;
                project.material_daemon = daemons::make_material_daemon();

                Scene basic = Scene::basic(context, project.material_daemon);
                project.scenes.push_back(basic);
                project.scenes_files.push_back(dir / "example.kobra");
                project.default_save_path = dir;
                project.default_scene_index = 0;

                return project;
        }
};

}
