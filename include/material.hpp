#ifndef KOBRA_MATERIAL_H_
#define KOBRA_MATERIAL_H_

// Standard headers
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>

// Engine headers
// #include "backend.hpp"
#include "common.hpp"
#include "core.hpp"
#include "types.hpp"

namespace kobra {

// Materials, in GGX form
struct Material {
	// TODO: default should be purple, flat shading
	glm::vec3	diffuse {1, 0, 1};
	glm::vec3	specular {0.0f};
	glm::vec3	emission {0.0f};
	glm::vec3	ambient {0.2f};
	float		shininess {0.0f};
	float		roughness {0.0f};
	float		refraction {1.0f};

	// TODO: extinction, absorption, etc.

	// TODO: refactor ttexture names (diffuse, etc)
	std::string	albedo_texture = "";
	std::string	normal_texture = "";
	std::string	specular_texture = "";
	std::string	emission_texture = "";
	std::string	roughness_texture = "";

	// TODO: emissive termm, reafctor eEmissive to eLight?
	Shading		type = Shading::eDiffuse;

	// Properties
	// TODO: refactor to better names...
	bool has_albedo() const;
	bool has_normal() const;
	bool has_specular() const;
	bool has_emission() const;
	bool has_roughness() const;

	// Serialize to GPU buffer
	void serialize(std::vector <aligned_vec4> &) const;
};

using MaterialPtr = std::shared_ptr <Material>;

}

#endif
