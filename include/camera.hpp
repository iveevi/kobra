#ifndef KOBRA_CAMERA_H_
#define KOBRA_CAMERA_H_

// Standard headers
#include <memory>

// Engine headers
#include "common.hpp"
#include "transform.hpp"
#include "vec.hpp"

namespace kobra {

// Camera properties are still affected by transform
struct Camera {
	Camera(float fov_ = 45.0f, float aspect_ = 1.0f)
			: fov(fov_), aspect(aspect_) {}

	// Perspective projection matrix
	glm::mat4 perspective_matrix() const {
		return glm::perspective(
			glm::radians(fov),
			aspect, 0.1f, 1000.0f
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

// Creating UVW frame
struct UVW {
	glm::vec3 u, v, w;
};

inline UVW uvw_frame(const Camera &camera, const Transform &transform, float aspect = -1.0f)
{
	glm::vec3 eye = transform.position;
	glm::vec3 lookat = eye + transform.forward();
	glm::vec3 up = transform.up();

	glm::vec3 w = lookat - eye;
	float wlen = glm::length(w);
	glm::vec3 u = glm::normalize(glm::cross(w, up));
	glm::vec3 v = glm::normalize(glm::cross(u, w));

	float vlen = wlen * glm::tan(glm::radians(camera.fov) / 2.0f);
	v *= vlen;

	float ulen = vlen * (aspect > 0 ? aspect : camera.aspect);
	u *= ulen;

	return {u, v, w};
}

using CameraPtr = std::shared_ptr <Camera>;

}

#endif
