#include "../include/camera.hpp"

// GLM
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

namespace mercury {

// Static variables
const glm::vec3 Camera::DEFAULT_UP(0.0f, 1.0f, 0.0f);

const float Camera::DEFAULT_YAW = -90.0f;
const float Camera::DEFAULT_PITCH = 0.0f;
const float Camera::DEFAULT_ZOOM = 45.0f;

// Constructors
Camera::Camera(const glm::vec3 &pos,
		const glm::vec3 &up,
		float eyaw,
		float epitch)
		: position(pos), world_up(up),
		yaw(eyaw), pitch(epitch),
		zoom(DEFAULT_ZOOM)
{
	_update_vecs();
}

// Private methods
void Camera::_update_vecs()
{
	front = glm::normalize(
		glm::vec3 {
			cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
			sin(glm::radians(pitch)),
			sin(glm::radians(yaw)) * cos(glm::radians(pitch)),
		}
	);

	right = glm::normalize(glm::cross(front, world_up));
	up = glm::normalize(glm::cross(right, front));
}

// Public methods
glm::mat4 Camera::get_view() const
{
	return glm::lookAt(position, position + front, up);
}

void Camera::move(const glm::vec3 &delta)
{
	position += delta;
}

}
