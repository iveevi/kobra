#ifndef KOBRA_MATERIAL_H_
#define KOBRA_MATERIAL_H_

// Standard headers
#include <cstdio>
#include <fstream>
#include <optional>

// Engine headers
#include "buffer_manager.hpp"
#include "core.hpp"
#include "sampler.hpp"
#include "types.hpp"
#include "common.hpp"

namespace kobra {

// Material
class Material {
	// Textures, if any
	Sampler 	*albedo_sampler = nullptr;
	Sampler 	*normal_sampler = nullptr;

	// Texture sources (for scene loading)
	std::string	albedo_source = "";
	std::string	normal_source = "";
public:
	glm::vec3	albedo;		// Albedo color
	float		shading_type;	// Shading types, see types.h
	float		ior;		// Index of refraction

	// Constructors
	Material(const glm::vec3 & = {1, 0, 1},
			float = SHADING_TYPE_SIMPLE,
			float = 1.0);

	// Copy constructor
	Material(const Material &);

	// Assignment operator
	Material &operator=(const Material &);

	// Destructor
	~Material();

	// Properties
	bool has_albedo() const;
	bool has_normal() const;

	// Get image descriptors
	std::optional <VkDescriptorImageInfo> get_albedo_descriptor() const;
	std::optional <VkDescriptorImageInfo> get_normal_descriptor() const;

	// Bind albdeo and normal textures
	void bind(const Vulkan::Context &,
			const VkCommandPool &,
			const VkDescriptorSet &, size_t, size_t) const;

	// Set textures
	void set_albedo(const Vulkan::Context &,
			const VkCommandPool &,
			const std::string &);

	void set_normal(const Vulkan::Context &,
			const VkCommandPool &,
			const std::string &);

	// Serialize to GPU buffer
	void serialize(Buffer4f *) const;

	// Save material to file
	void save(std::ofstream &) const;

	// Read material from file
	// TODO: pack args into a struct
	static std::optional <Material> from_file(const Vulkan::Context &,
			const VkCommandPool &,
			std::ifstream &,
			const std::string &);
};

}

#endif
