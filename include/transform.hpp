#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Standard headers
#include <fstream>
#include <iostream>
#include <optional>

// Engine headers
#include "vec.hpp"

namespace kobra {

// Transform class
struct Transform {
	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 scale;

	// Constructor
	Transform(const glm::vec3 &p = glm::vec3 {0.0f},
		const glm::vec3 &r = glm::vec3 {0.0f},
		const glm::vec3 &s = glm::vec3 {1.0f})
			: position(p), rotation(r), scale(s) {}

	// Copy constructor
	Transform(const Transform &t)
			: position(t.position), rotation(t.rotation),
			scale(t.scale) {}

	// Calculate the model matrix
	glm::mat4 matrix() const {
		glm::mat4 pmat = glm::translate(glm::mat4(1.0f), position);
		glm::mat4 rmat = glm::mat4_cast(glm::quat(glm::radians(rotation)));
		glm::mat4 smat = glm::scale(glm::mat4(1.0f), scale);
		return pmat * rmat * smat;
	}

	// Apply transform on a point
	// TODO: refactor
	glm::vec3 apply(const glm::vec3 &v) const {
		glm::mat4 m = matrix();
		return glm::vec3(m * glm::vec4(v, 1.0f));
	}

	// Apply transform on a vector
	glm::vec3 apply_vector(const glm::vec3 &v) const {
		glm::mat4 m = matrix();
		return glm::vec3(m * glm::vec4(v, 0.0f));
	}

	// Move the transform
	void move(const glm::vec3 &delta) {
		position += delta;
	}

	// Look in a direction
	void look(const glm::vec3 &direction) {
		rotation = glm::eulerAngles(glm::quatLookAt(direction, glm::vec3(0.0f, 1.0f, 0.0f)));
	}

	// Get forward, right, up vectors
	glm::vec3 forward() const {
		glm::quat q = glm::quat(rotation);
		return glm::normalize(glm::vec3(q * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
	}

	glm::vec3 right() const {
		glm::quat q = glm::quat(rotation);
		return glm::normalize(glm::vec3(q * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
	}

	glm::vec3 up() const {
		glm::quat q = glm::quat(rotation);
		return glm::normalize(glm::vec3(q * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
	}

	// Arithmetic
	glm::vec3 operator*(const glm::vec3 &v) const {
		return apply(v);
	}

	// Boolean operators
	bool operator==(const Transform &t) const {
		return position == t.position
			&& rotation == t.rotation
			&& scale == t.scale;
	}

	bool operator!=(const Transform &t) const {
		return !(*this == t);
	}
};

inline std::ostream &operator<<(std::ostream &os, const Transform &t)
{
	return os << glm::to_string(t.matrix());
}

}

#endif
