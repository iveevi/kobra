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
        Transform() {}
        Transform(const glm::vec3 &pos) : position {pos},
                rotation {1.0, 0.0, 0.0, 0.0},
                scale {1.0, 1.0, 1.0},
                forward {0.0, 0.0, 1.0},
                up {0.0, 1.0, 0.0},
                right {1.0, 0.0, 0.0} {}

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
