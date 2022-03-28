#ifndef SCENE_H_
#define SCENE_H_

// Standard headers
#include <fstream>
#include <string>
#include <vector>

// Engine headers
#include "logger.hpp"
#include "object.hpp"

namespace kobra {

// Scene stores objects
class Scene {
	std::vector <ObjectPtr> _objects;
public:
	// Default constructor
	Scene() = default;

	// Constructor from file
	Scene(const std::string &);

	// Constructor from list of objects
	Scene(const std::vector <Object *> &);
	Scene(const std::vector <ObjectPtr> &);

	// Save objects to file
	void save(const std::string &) const;
};

}

#endif
