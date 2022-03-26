#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Standard headers
#include <iostream>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Transform class
class Transform {
	glm::mat4 model;
	glm::mat4 inv;

	void _calc_inv() {
		inv = glm::inverse(model);
	}
public:
	// Default constructor
	Transform() : model(glm::mat4(1.0f)), inv(glm::mat4(1.0f)) {}

	// Constructor
	Transform(const glm::mat4 &m) : model {m} {
		_calc_inv();
	}

	Transform(const glm::vec3 &p) : model {
			glm::translate(glm::mat4(1.0f), p)
		} {
		_calc_inv();
	}

	// Move
	void move(const glm::vec3 &p) {
		model = glm::translate(model, p);
		_calc_inv();
	}

	// Rotate
	void rotate(const glm::vec3 &axis, float angle) {
		model = glm::rotate(model, angle, axis);
		_calc_inv();
	}

	// Rotate euler angles
	void rotate(const glm::vec3 &angles) {
		model = glm::rotate(model, angles.x, glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, angles.y, glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, angles.z, glm::vec3(0.0f, 0.0f, 1.0f));
		_calc_inv();
	}

	// Rotate in place
	void rotate_in_place(const glm::vec3 &axis, float angle) {
		glm::vec3 p = glm::vec3(model[3]);
		model = glm::translate(model, -p);
		model = glm::rotate(model, angle, axis);
		model = glm::translate(model, p);
		_calc_inv();
	}

	// Scale
	void scale(const glm::vec3 &s) {
		model = glm::scale(model, s);
		_calc_inv();
	}

	void scale(float s) {
		model = glm::scale(model, glm::vec3(s));
		_calc_inv();
	}

	// Turn to face
	/* void lookat(const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up = {0.0f, 1.0f, 0.0f}) {
		model = glm::lookAt(eye, target, up);
	} */

	// Get the model matrix
	const glm::mat4 &matrix() const {
		return model;
	}

	// Set the model matrix
	void set_matrix(const glm::mat4 &m) {
		model = m;
		_calc_inv();
	}

	// Get position
	glm::vec3 position() const {
		return glm::vec3(model[3]);
	}

	// Get rotation
	glm::quat rotation() const {
		return glm::quat_cast(model);
	}

	// Set position
	void set_position(const glm::vec3 &p) {
		model[3][0] = p.x;
		model[3][1] = p.y;
		model[3][2] = p.z;
		_calc_inv();
	}

	// Forward, right, up
	glm::vec3 forward() const {
		return glm::normalize(
			glm::vec3(inv[2])
		);
	}

	glm::vec3 right() const {
		return glm::normalize(
			glm::vec3(inv[0])
		);
	}

	glm::vec3 up() const {
		return glm::normalize(
			glm::vec3(inv[1])
		);
	}
};

#endif
