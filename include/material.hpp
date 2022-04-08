#ifndef MATERIAL_H_
#define MATERIAL_H_

// Standard headers
#include <cstdio>
#include <fstream>
#include <optional>

// Engine headers
#include "buffer_manager.hpp"
#include "core.hpp"
#include "texture.hpp"
#include "types.hpp"
#include "world_update.hpp"
#include "common.hpp"

namespace kobra {

// Material
struct Material {
	// Shading type
	glm::vec3	albedo = glm::vec3 {1.0, 0.0, 1.0};

	float		shading_type = SHADING_TYPE_SIMPLE;
	float		ior = 1.0;

	// Sources
	std::string	albedo_source;
	std::string	normal_source;

	// Texture
	// TODO: this should be privagte
	Sampler 	*albedo_sampler = nullptr;
	Sampler 	*normal_sampler = nullptr;

	// Setters
	void set_albedo(const Vulkan::Context &ctx, const VkCommandPool &command_pool, const std::string &path) {
		albedo_source = path;
		albedo_sampler = new Sampler(ctx, command_pool, path);
	}

	void set_normal(const Vulkan::Context &ctx, const VkCommandPool &command_pool, const std::string &path) {
		normal_source = path;
		normal_sampler = new Sampler(ctx, command_pool, path);
	}

	void write_material(Buffer4f *bf_mats) const {
		bf_mats->push_back(aligned_vec4(albedo, shading_type));
		bf_mats->push_back(aligned_vec4(
			{ior, (!albedo_sampler), (!normal_sampler), 0}
		));
	}

	// Save material to file
	void save(std::ofstream &file) const {
		char buf[1024];
		sprintf(buf, "%d", *((int *) &shading_type));

		file << "[MATERIAL]\n";
		file << "albedo=" << albedo.x << " " << albedo.y << " " << albedo.z << std::endl;
		file << "shading_type=" << buf << std::endl;
		file << "ior=" << ior << std::endl;

		file << "albedo_source=" << (albedo_sampler ? albedo_source : "0") << std::endl;
		file << "normal_source=" << (normal_sampler ? normal_source : "0") << std::endl;
	}

	// Read material from file
	// TODO: pack args into a struct
	static std::optional <Material> from_file(const Vulkan::Context &ctx,
			const VkCommandPool &command_pool,
			std::ifstream &file,
			const std::string &scene_file) {
		std::string line;

		// Read albedo
		glm::vec3 albedo;
		std::getline(file, line);
		std::sscanf(line.c_str(), "albedo=%f %f %f", &albedo.x, &albedo.y, &albedo.z);

		// Read shading type
		char buf1[1024];
		float shading_type;
		int tmp;
		std::getline(file, line);
		std::sscanf(line.c_str(), "shading_type=%s", buf1);
		
		// TODO: helper method
		// Check if the shading type is literal
		std::string stype = buf1;
		if (stype == "DIFFUSE") {
			shading_type = SHADING_TYPE_DIFFUSE;
		} else if (stype == "EMISSIVE") {
			shading_type = SHADING_TYPE_EMISSIVE;
		} else if (stype == "REFLECTION") {
			shading_type = SHADING_TYPE_REFLECTION;
		} else if (stype == "REFRACTION") {
			shading_type = SHADING_TYPE_REFRACTION;
		} else {
			std::sscanf(buf1, "%d", &tmp);
			shading_type = tmp;
			shading_type = *((float *) &tmp);
		}

		// Read ior
		float ior;
		std::getline(file, line);
		std::sscanf(line.c_str(), "ior=%f", &ior);

		// Read albedo source
		char buf2[1024];
		std::string albedo_source;
		std::getline(file, line);
		std::sscanf(line.c_str(), "albedo_source=%s", buf2);
		albedo_source = buf2;

		// Read normal source
		char buf3[1024];
		std::string normal_source;
		std::getline(file, line);
		std::sscanf(line.c_str(), "normal_source=%s", buf3);
		normal_source = buf3;

		// Construct and return material
		Material mat;
		mat.albedo = albedo;
		mat.shading_type = shading_type;
		mat.ior = ior;

		if (albedo_source != "0") {
			albedo_source = common::get_path(
				albedo_source,
				common::get_directory(scene_file)
			);

			mat.set_albedo(ctx, command_pool, albedo_source);
		}

		if (normal_source != "0") {
			normal_source = common::get_path(
				normal_source,
				common::get_directory(scene_file)
			);

			mat.set_normal(ctx, command_pool, normal_source);
		}

		// Return material
		return mat;
	}
};

}

#endif
