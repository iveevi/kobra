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
		float yaw,
		float pitch)
		: position(pos), world_up(up),
		_yaw(yaw), _pitch(pitch),
		zoom(DEFAULT_ZOOM)
{
	_update_vecs();
}

// Private methods
void Camera::_update_vecs()
{
	front = glm::normalize(
		glm::vec3 {
			cos(glm::radians(_yaw)) * cos(glm::radians(_pitch)),
			sin(glm::radians(_pitch)),
			sin(glm::radians(_yaw)) * cos(glm::radians(_pitch)),
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

void Camera::set_yaw(float yaw)
{
	_yaw = yaw;
	_update_vecs();
}

void Camera::set_pitch(float pitch, bool constrain)
{
	_pitch = pitch;

	// Clamp the pitch
	if (constrain)
		_pitch = std::fmax(-89.0f, std::fmin(89.0, _pitch));

	_update_vecs();
}

void Camera::add_yaw(float dyaw)
{
	set_yaw(_yaw + dyaw);
}

void Camera::add_pitch(float dpitch, bool constrain)
{
	set_pitch(_pitch + dpitch);
}

void Camera::move(const glm::vec3 &delta)
{
	position += delta;
}

}
