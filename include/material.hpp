#ifndef MATERIAL_H_
#define MATERIAL_H_

// Standard headers
#include <fstream>

// Engine headers
#include "buffer_manager.hpp"
#include "core.hpp"
#include "types.hpp"
#include "world_update.hpp"

namespace kobra {

// Material
struct Material {
	// Shading type
	glm::vec3 albedo	= glm::vec3 {1.0, 0.0, 1.0};

	float reflectance	= 0.0f;
	float refractance	= 0.0f;
	float extinction	= 0.0f;

	// Write to buffer
	// TODO: delete this
	void write_material(kobra::WorldUpdate &wu) const {
		wu.bf_mats->push_back(aligned_vec4(albedo, SHADING_TYPE_BLINN_PHONG));
		wu.bf_mats->push_back(aligned_vec4(
			{0, reflectance, refractance, extinction}
		));
	}

	void write_material(Buffer4f *bf_mats) const {
		bf_mats->push_back(aligned_vec4(albedo, SHADING_TYPE_BLINN_PHONG));
		bf_mats->push_back(aligned_vec4(
			{0, reflectance, refractance, extinction}
		));
	}

	// Save material to file
	void save(std::ofstream &file) const {
		file << "[MATERIAL]\n";
		file << "albedo=" << albedo.x << " " << albedo.y << " " << albedo.z << std::endl;
		file << "reflectance=" << reflectance << std::endl;
		file << "refractance=" << refractance << std::endl;
		file << "extinction=" << extinction << std::endl;
	}
};

}

#endif
