#pragma once

// Standard headers
#include <string>

// Engine headers
#include "backend.hpp"
#include "ecs.hpp"

namespace kobra {

// Scene class
struct Scene {
	std::shared_ptr <ECS> ecs;

	// Other scene-local data
	std::string p_environment_map;

	// Saving and loading
	void save(const std::string &);
	void load(const Context &, const std::string &);

	// Populate cache list of meshes
	void populate_mesh_cache(std::set <MeshPtr> &mesh_cache) const {
		ecs->populate_mesh_cache(mesh_cache);
	}
};

}
