#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Standard headers
#include <fstream>
#include <iostream>
#include <optional>

// GLM headers
#include <glm/gtx/matrix_decompose.hpp>

// Engine headers
#include "vec.hpp"

namespace kobra {

// Transform class
struct Transform {
	glm::vec3 position;
	glm::vec3 rotation; // in degrees
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

        // Obtain proeprties from matrix
        Transform(const glm::mat4 &mat) {
                position = mat[3];

                scale.x = glm::length(glm::vec3(mat[0])); // Basis vector X
                scale.y = glm::length(glm::vec3(mat[1])); // Basis vector Y
                scale.z = glm::length(glm::vec3(mat[2])); // Basis vector Z

                const glm::vec3 left	= glm::normalize(glm::vec3(mat[0])); // Normalized left axis
                const glm::vec3 up	= glm::normalize(glm::vec3(mat[1])); // Normalized up axis
                const glm::vec3 forward	= glm::normalize(glm::vec3(mat[2])); // Normalized forward axis

                // Obtain the "unscaled" transform matrix
                glm::mat4 m(0.0f);
                m[0][0] = left.x;
                m[0][1] = left.y;
                m[0][2] = left.z;

                m[1][0] = up.x;
                m[1][1] = up.y;
                m[1][2] = up.z;

                m[2][0] = forward.x;
                m[2][1] = forward.y;
                m[2][2] = forward.z;

                rotation.x = atan2f( m[1][2], m[2][2]);
                rotation.y = atan2f(-m[0][2], sqrtf(m[1][2] * m[1][2] + m[2][2] * m[2][2]));
                rotation.z = atan2f( m[0][1], m[0][0]);
                rotation = glm::degrees(rotation); // Convert to degrees, or you could multiply it by (180.f / 3.14159265358979323846f)
        }

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
