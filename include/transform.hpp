#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Standard headers
#include <iostream>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Transform class
struct Transform {
        // Basic properties
        glm::vec3 position;
        glm::quat rotation;
        glm::vec3 scale;

        // Directions
        glm::vec3 forward;
        glm::vec3 up;
        glm::vec3 right;

        // Constructors
        Transform() : Transform({0.0, 0.0, 0.0}) {}
        Transform(const glm::vec3 &pos) : position {pos},
                rotation {1.0, 0.0, 0.0, 0.0},
                scale {1.0, 1.0, 1.0},
                forward {0.0, 0.0, 1.0},
                up {0.0, 1.0, 0.0},
                right {1.0, 0.0, 0.0} {}
	
	// Full constructor
	Transform(const glm::vec3 &pos, const glm::quat &rot, const glm::vec3 s,
			const glm::vec3 &f, const glm::vec3 &u, const glm::vec3 &r)
			: position {pos}, rotation {rot}, scale {s},
			forward(f), up(u), right(r) {}

	void recalculate() {
		// Cardinal Directions
		// TODO: static
		glm::vec4 f = {0.0, 0.0, 1.0, 0.0};
		glm::vec4 u = {0.0, 1.0, 0.0, 0.0};
		glm::vec4 r = {1.0, 0.0, 0.0, 0.0};

		forward = glm::normalize(glm::vec3(rotation * f));
		up = glm::normalize(glm::vec3(rotation * u));
		right = glm::normalize(glm::vec3(rotation * r));
	}

	// Get as mat4
	// TODO: cache
	glm::mat4 model() const {
		glm::mat4 model {1.0};
		model = glm::translate(model, position);
		model = glm::scale(model, scale);
		model = glm::mat4_cast(rotation) * model;
		return model;
	}

	// Apply transformation to vec3
	glm::vec3 apply(const glm::vec3 &v) const {
		return glm::vec3(model() * glm::vec4(v, 1.0));
	}

	// Set pitch
	void set_pitch(float pitch) {
		rotation = glm::angleAxis(pitch, right);
		recalculate();
	}

	// Set yaw
	void set_yaw(float yaw) {
		rotation = glm::angleAxis(yaw, up);
		recalculate();
	}

	// Set euler angles
	void set_euler(float pitch, float yaw, float roll = 0.0f) {
		rotation = glm::quat(glm::vec3(pitch, yaw, roll));
		recalculate();
	}
};

#endif
