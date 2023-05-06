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

        // Default scene
        static Scene basic(const Context &context) {
                // TODO: add camera, plane, and 2 boxes...

                Scene scene;
                scene.name = "Example";
                scene.ecs = std::make_shared <ECS> ();

                // Add a plane
                Mesh plane = Mesh::plane();

                // Manually allocate materials
                Material::all.clear();
                Material plane_material;

                int index = Material::all.size();
                Material::all.push_back(plane_material);

                for (auto &submesh : plane.submeshes)
                        submesh.material_index = index;
               
                // Create the plane entity
                Entity entity;
                entity = scene.ecs->make_entity("Plane");
                entity.add <Mesh> (plane);

                Mesh *mesh = &entity.get <Mesh> ();
                entity.add <Renderable> (context, mesh);

                return scene;
        }
};

}
