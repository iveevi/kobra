#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

// Transform struct
struct Transform {
	glm::vec3	translation;
	glm::vec3	erot;		// Euler angles
	glm::vec3	scale;
	// Constructors
	Transform();		// Identity transform
	Transform(const glm::vec3 &, const glm::vec3 & = {0, 0, 0},
			const glm::vec3 & = {1, 1, 1});

	// Methods
	void move(const glm::vec3 &);
	void rotate(const glm::vec3 &);

	glm::mat4 model() const;
};

}

#endif