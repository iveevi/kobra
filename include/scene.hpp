#ifndef SCENE_H_
#define SCENE_H_

// Standard headers
#include <fstream>
#include <string>
#include <vector>

// Engine headers
#include "backend.hpp"
#include "logger.hpp"
#include "object.hpp"

namespace kobra {

// Scene stores objects
class Scene {
public:
	// Iterators
	using iterator = std::vector <ObjectPtr> ::const_iterator;
private:
	std::vector <ObjectPtr> _objects;

	// TODO: hash map of names
public:
	// Default constructor
	Scene() = default;

	// Constructor from file
	Scene(const Vulkan::Context &,
		const VkCommandPool &,
		const std::string &);

	// Constructor from list of objects
	Scene(const std::vector <Object *> &);
	Scene(const std::vector <ObjectPtr> &);

	// Iterators
	iterator begin() const;
	iterator end() const;

	// Retrieve object by name
	ObjectPtr operator[](const std::string &) const;

	// Save objects to file
	void save(const std::string &) const;
};

}

#endif
