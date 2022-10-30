#ifndef KOBRA_PROJECT_H_
#define KOBRA_PROJECT_H_

// Standard headers
#include <filesystem>
#include <fstream>
#include <string>

namespace kobra {

// Project description; for a complete game, rendering application, etc.
struct Project {
	// TODO: scenes as a list that is lazily loaded...
	std::string scene;

	// Loading projects
	static Project load(const std::string &path) {
		// Check if the file exists
		if (!std::filesystem::exists(path)) // TODO: kobra error...
			throw std::runtime_error("Project file does not exist");

		std::ifstream file(path);

		Project project;
		file >> project.scene;

		return project;
	}
};

}

#endif
