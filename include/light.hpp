#ifndef LIGHT_H_
#define LIGHT_H_

#include "object.hpp"

// Base class for all lights
struct Light : public Object {
	// Constructor is a position
	Light(const glm::vec3 &position)
		: Object(position) {}

	// Virtual destructor
	virtual ~Light() {}

	////////////////////////////////////////////
	// Virtual functions for light properties //
	////////////////////////////////////////////
	
	// Diffuse value from point and normal
	virtual float diffuse(const glm::vec3 &, const glm::vec3 &) const = 0;
};

struct PointLight : public Light {
	// TODO: color
 
	// For now, just a position
        PointLight(const glm::vec3 &position)
		: Light(position) {}

	// Diffuse value for point light
	float diffuse(const glm::vec3 &p, const glm::vec3 &n) const {
		// Normalize the normal and the light position delta
		glm::vec3 nn = glm::normalize(n);
		glm::vec3 nd = glm::normalize(p - position);

		// Return diffuse value
		return glm::max(0.0f, glm::dot(nn, nd));
	}
};

#endif
