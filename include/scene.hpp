#ifndef SCENE_H_
#define SCENE_H_

// Standard headers
#include <fstream>
#include <string>

// Engine headers
#include "logger.hpp"
#include "world.hpp"

namespace kobra {

// Scene
//	will eventually take care of
//	more than just loading and
//	saving the world
class Scene {
	// For now only contains name and world
	std::string	_name;

	World		_world;
public:
	// Constructor
	Scene();
	Scene(const std::string &str, const World &w)
			: _name(str), _world(w) {}	// TODO: remove

	// Save to file
	void save(const std::string &filename) const {
		// Open the file
		std::ofstream file(filename);

		// Check if the file is open
		if (!file.is_open()) {
			Logger::error() << "Failed to open file: "
				<< filename << "\n";
			return;
		}

		// Write name as header
		file << "===[SCENE: " << _name << "]===\n";

		// Iterate through each object
		for (const auto &prim : _world.objects)
			prim->save_to_file(file);

		// Log
		Logger::ok() << "Successfully saved scene \"" << _name
			<< "\" to file: " << filename << "\n";
	}
};

}

#endif