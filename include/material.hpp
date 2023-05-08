#pragma once

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
        // Name of the material
        std::string name;

	// TODO: default should be purple, flat shading
	glm::vec3	diffuse {1, 0, 1};
	glm::vec3	specular {0.0f};
	glm::vec3	emission {0.0f};
	glm::vec3	ambient {0.2f}; // TODO: deprecate
	float		shininess {0.0f}; // TODO: deprecate
	float		roughness {0.001f};
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

	// TODO: each material needs a reference to a shader program...
	// how do we extend it to support multiple shader languages, like CUDA?
	// Material presets:
	// 	GGX, Disney, Blinn-Phong based materials?
	// 	Then restrict CUDA to GGX only?
	//	Actually cuda can implement all these, and then extract
	//	appropriate during calculate_material()

	// Properties
	// TODO: refactor to better names...
	bool has_albedo() const;
	bool has_normal() const;
	bool has_specular() const;
	bool has_emission() const;
	bool has_roughness() const;

	// Serialize to GPU buffer
	void serialize(std::vector <aligned_vec4> &) const;

	// Construct the default material
	// NOTE: different from the uninitialized material
	static Material default_material() {
                // TODO: mathod in the material daemon instead...
		Material mat;
		mat.name = "default";
		mat.diffuse = glm::vec3 {1};
		mat.specular = glm::vec3 {0.5f};
		mat.roughness = 0.2f;
		return mat;
	}
};

}
