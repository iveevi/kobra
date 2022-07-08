#ifndef KOBRA_CAMERA_H_
#define KOBRA_CAMERA_H_

// Standard headers
#include <memory>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Engine headers
#include "common.hpp"
#include "transform.hpp"

namespace kobra {

// TODO: clean up this interface

// Camera properties
class Tunings {
        float _radians(float degrees) {
                return degrees * (M_PI / 180.0f);
        }
public:
        float fov;
        float scale;
        float aspect;

        // TODO: add lens attributes
        Tunings() : fov(90.0f), aspect(1.0f) {
                scale = tan(_radians(fov) * 0.5f);
        }

        Tunings(float fov, float width, float height)
                        : fov(fov), aspect(width / height) {
                scale = tan(_radians(fov) * 0.5f);
        }
};

// Camera class
// TODO: should be object type
struct Camera {
        // Camera properties
        Tunings tunings;

	// Public member variables
        Transform transform;

        // Constructors
        Camera() {}
        Camera(const Transform& trans, const Tunings& tns)
                        : transform(trans), tunings(tns) {}

	// Get view matrix
	glm::mat4 view() const {
		return glm::lookAt(
			transform.position,
			transform.position + transform.forward(),
			transform.up()
		);
	};

	// Get perspective matrix
	glm::mat4 perspective() const {
		return glm::perspective(
			glm::radians(tunings.fov),
			tunings.aspect,
			0.01f, 100.0f
		);

	}

	// Get projection matrix
	glm::mat4 projection() const {
		return perspective() * view();
	}

	// Generate ray
	Ray generate_ray(float x, float y) const {
		float cx = (2 * x - 1) * tunings.scale * tunings.aspect;
		float cy = (1 - 2 * y) * tunings.scale;

		glm::vec3 dir = cx * transform.right()
			+ cy * transform.up()
			+ transform.forward();

		return Ray {transform.position, dir};
	}
};

using CameraPtr = std::shared_ptr <Camera>;

}

#endif
