#ifndef KOBRA_SCENE_H_
#define KOBRA_SCENE_H_

// Standard headers
#include <string>

// Engine headers
#include "backend.hpp"
#include "ecs.hpp"

namespace kobra {

// Scene class
struct Scene {
	ECS ecs;

	// Other scene-local data
	std::string p_environment_map;

	// Saving and loading
	void save(const std::string &);
	void load(const Context &, const std::string &);
};

}

#endif
