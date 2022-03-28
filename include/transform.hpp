#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Standard headers
#include <iostream>

// GLM headers
#define GLM_PERSPECTIVE_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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
		glm::mat4 model = glm::mat4(1.0f);
		
		glm::mat4 pmat = glm::translate(glm::mat4(1.0f), position);
		glm::mat4 rmat = glm::mat4_cast(glm::quat(glm::radians(rotation)));
		glm::mat4 smat = glm::scale(glm::mat4(1.0f), scale);

		return pmat * rmat * smat;
	}

	// Move the transform
	void move(const glm::vec3 &delta) {
		position += delta;
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
};

/* Transform class
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
		glm::vec3 pos = model[3];
		model = glm::mat4(1.0f);
		model = glm::rotate(model, angles.x, glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, angles.y, glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, angles.z, glm::vec3(0.0f, 0.0f, 1.0f));
		model = glm::translate(model, pos);
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

	// Set euler angles
	void set_angles(const glm::vec3 &angles) {
		model = glm::rotate(model, angles.x, glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, angles.y, glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, angles.z, glm::vec3(0.0f, 0.0f, 1.0f));
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

	// Look at
	void look_at(const glm::vec3 &target, const glm::vec3 &up = {0.0f, 1.0f, 0.0f}) {
		model = glm::lookAt(glm::vec3(model[3]), target, up);
	} 

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
}; */

}

#endif
