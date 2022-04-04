#ifndef MATERIAL_H_
#define MATERIAL_H_

// Standard headers
#include <fstream>

// Engine headers
#include "buffer_manager.hpp"
#include "core.hpp"
#include "texture.hpp"
#include "types.hpp"
#include "world_update.hpp"

namespace kobra {

// Material
struct Material {
	// TODO: possible some samplers as well

	// Shading type
	glm::vec3	albedo = glm::vec3 {1.0, 0.0, 1.0};

	float		shading_type = SHADING_TYPE_SIMPLE;
	float		ior = 1.0;

	// Texture
	// TODO: this should be privagte
	Sampler 	*albedo_sampler = nullptr;

	// Setters
	void set_albedo(const Vulkan::Context &ctx, const VkCommandPool &command_pool, const std::string &path) {
		albedo_sampler = new Sampler(ctx, command_pool, path);
	}

	// Write to buffer
	// TODO: delete this
	void write_material(kobra::WorldUpdate &wu) const {
		wu.bf_mats->push_back(aligned_vec4(albedo, shading_type));
		wu.bf_mats->push_back(aligned_vec4(
			{ior, (!albedo_sampler), 0, 0}
		));
	}

	void write_material(Buffer4f *bf_mats) const {
		bf_mats->push_back(aligned_vec4(albedo, shading_type));
		bf_mats->push_back(aligned_vec4(
			{ior, (!albedo_sampler), 0, 0}
		));
	}

	// Save material to file
	void save(std::ofstream &file) const {
		file << "[MATERIAL]\n";
		file << "albedo=" << albedo.x << " " << albedo.y << " " << albedo.z << std::endl;
		file << "ior=" << ior << std::endl;
	}
};

}

#endif
