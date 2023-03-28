#pragma once

// Standard headers
#include <string>

// Engine headers
#include "backend.hpp"
#include "ecs.hpp"
#include "mesh.hpp"

namespace kobra {

// Scene class
struct Scene {
	std::string name;
	std::shared_ptr <ECS> ecs;

	// Other scene-local data
	std::string p_environment_map;

	// Saving and loading
	void save(const std::string &);
	void load(const Context &, const std::string &);

	// Populate cache list of meshes
	void populate_mesh_cache(std::set <const Submesh *> &submesh_cache) const {
		ecs->populate_mesh_cache(submesh_cache);
	}
};

}
