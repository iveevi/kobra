#ifndef MATERIAL_H_
#define MATERIAL_H_

// Engine headers
#include "core.hpp"
#include "types.h"

// Material
struct Material {
	// Shading type
	float shading = SHADING_TYPE_BLINN_PHONG;

	// For now, just a color
	glm::vec3 color;

	Material() {}
	Material(const glm::vec3 &c) : color(c) {}
	Material(const glm::vec3 &c, float s) : shading(s), color(c) {}

	// Write to buffer
	void write_to_buffer(Buffer &buffer) const {
		buffer.push_back(aligned_vec4(color, shading));
	}
};

#endif
