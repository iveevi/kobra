#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

// Engine headers
#include "include/math/compat.hpp"

namespace mercury {

// Transform struct
struct Transform {
	vec3	translation;
	// glm::vec3	erot;		// Euler angles
	vec3	scale;
	quat	orient;

	// Constructors
	Transform();		// Identity transform
	Transform(const vec3 &, const vec3 & = {0, 0, 0},
			const vec3 & = {1, 1, 1});

	// Methods
	void move(const vec3 &);
	void rotate(const vec3 &);
	void rotate(const quat &);

	glm::mat4 model() const;
};

}

#endif