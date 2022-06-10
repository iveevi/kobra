#ifndef KOBRA_MATERIAL_H_
#define KOBRA_MATERIAL_H_

// Standard headers
#include <cstdio>
#include <fstream>
#include <optional>

// Engine headers
#include "backend.hpp"
#include "common.hpp"
#include "core.hpp"
#include "types.hpp"
// #include "buffer_manager.hpp"
// #include "sampler.hpp"

namespace kobra {

// Material
class Material {
	// Textures, if any
	vk::raii::Sampler albedo_sampler = nullptr;
	vk::raii::Sampler normal_sampler = nullptr;

	ImageData albedo_image = nullptr;
	ImageData normal_image = nullptr;

	// Source device
	vk::raii::Device *device = nullptr;

	// Texture sources (for scene loading)
	// TODO: Kd, Kd, normal
	std::string	albedo_source = "";
	std::string	normal_source = "";
public:
	glm::vec3	Kd = glm::vec3(0.0f);
	glm::vec3	Ks = glm::vec3(0.0f);

	// TODO: default should be purple, no shading
	Shading		type = Shading::eDiffuse;

	float		refr_eta = 1.0f;
	float		refr_k = 0.0f;


	// Defualt constructor
	Material() = default;

	// No copies or assignments
	Material(const Material &) = delete;
	Material &operator=(const Material &) = delete;

	// Moves only
	Material(Material &&) = default;
	Material &operator=(Material &&) = default;

	// Create a deep copy of the material
	Material &&copy() const {
		Material m;

		/* Create copies for each texture only if they exist
		if (albedo_source != "") {
			// Create duplicate image data
			// TODO: this is wasteful

		} */

		// TODO: fill
		m.Kd = glm::vec3 {1.0, 0.0, 1.0}; // dummy color

		// Return the moved material
		return std::move(m);
	}

	// Properties
	bool has_albedo() const;
	bool has_normal() const;

	// Get image descriptors
	std::optional <vk::DescriptorImageInfo> get_albedo_descriptor() const;
	std::optional <vk::DescriptorImageInfo> get_normal_descriptor() const;

	// Bind albdeo and normal textures
	void bind(const vk::raii::Device &, const vk::raii::DescriptorSet &, uint32_t, uint32_t) const;

	// Set textures
	void set_albedo(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const vk::raii::CommandPool &,
			const std::string &);

	void set_normal(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const vk::raii::CommandPool &,
			const std::string &);

	// Serialize to GPU buffer
	void serialize(std::vector <aligned_vec4> &) const;

	// Save material to file
	void save(std::ofstream &) const;

	// Read material from file
	// TODO: pack args into a struct
	static Material &&from_file
			(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const vk::raii::CommandPool &,
			std::ifstream &,
			const std::string &,
			bool &);
};

}

#endif
