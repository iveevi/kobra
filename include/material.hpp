#ifndef MATERIAL_H_
#define MATERIAL_H_

// Engine headers
#include "core.hpp"
#include "types.hpp"

// Material
struct Material {
	// Shading type
	glm::vec3 albedo;
	float shading		= SHADING_TYPE_BLINN_PHONG;

	float specular		= 32.0f;
	float reflectance	= 0.0f;
	float refraction	= 0.0f;

	// Write to buffer
	void write_to_buffer(Buffer &buffer) const {
		buffer.push_back(aligned_vec4(albedo, shading));
		buffer.push_back(aligned_vec4(
			{specular, reflectance, refraction, 0.0f}
		));
	}
};

#endif
