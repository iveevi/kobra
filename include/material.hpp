#ifndef KOBRA_MATERIAL_H_
#define KOBRA_MATERIAL_H_

// Standard headers
#include <cstdio>
#include <fstream>
#include <optional>

// Engine headers
// #include "backend.hpp"
#include "common.hpp"
#include "core.hpp"
#include "types.hpp"

namespace kobra {

struct Material {
	// TODO: default should be purple, flat shading
	std::string	albedo_source = "";
	std::string	normal_source = "";

	glm::vec3	Kd {1, 0, 1};
	glm::vec3	Ks {0.0f};

	// TODO: emissive termm, reafctor eEmissive to eLight?
	Shading		type = Shading::eDiffuse;

	float		refr_eta = 1.0f;
	float		refr_k = 0.0f;

	// Properties
	bool has_albedo() const;
	bool has_normal() const;

	// Save material to file
	void save(std::ofstream &) const;

	// Serialize to GPU buffer
	void serialize(std::vector <aligned_vec4> &) const;

	static Material from_file(std::ifstream &, const std::string &, bool &);
};

// TODO: eventually use GGX for roughness

}

#endif
