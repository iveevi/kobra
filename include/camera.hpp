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

// Camera properties are still affected by transform
struct Camera {
	Camera(float fov_ = 45.0f, float aspect_ = 1.0f)
			: fov(fov_), aspect(aspect_) {}

	// Perspective projection matrix
	glm::mat4 perspective_matrix() const {
		return glm::perspective(
			glm::radians(fov),
			aspect, 0.1f, 100.0f
		);
	}

	// View matrix
	static glm::mat4 view_matrix(const Transform &transform) {
		return glm::lookAt(
			transform.position,
			transform.position + transform.forward(),
			transform.up()
		);
	}
	
	float		fov;
	float		aspect;
};

using CameraPtr = std::shared_ptr <Camera>;

}

#endif
