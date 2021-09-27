#ifndef CAMERA_H_
#define CAMERA_H_

// GLM
#include <glm/fwd.hpp>
#include <glm/glm.hpp>

namespace mercury {

class Camera {
	void _update_vecs();

	// Camera rotation
	float _yaw;
	float _pitch;
public:
	// TODO: private any attributes?
	// Camera position and orientation
	glm::vec3 position;

	// TODO: should this be a struct
	// global orientation?
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;

	glm::vec3 world_up;

	// Other options
	float zoom;

	// Constructors
	Camera(const glm::vec3 & = DEFAULT_POS,
		const glm::vec3 & = DEFAULT_UP,
		float = DEFAULT_YAW,
		float = DEFAULT_PITCH);

	// Getters
	glm::mat4 get_view() const;

	// Setters
	void set_yaw(float);
	void set_pitch(float, bool = true);

	void add_yaw(float);
	void add_pitch(float, bool = true);

	// Methods
	void move(const glm::vec3 &);

	// Static variables
	static const glm::vec3 DEFAULT_POS;
	static const glm::vec3 DEFAULT_UP;

	static const float DEFAULT_YAW;
	static const float DEFAULT_PITCH;
	static const float DEFAULT_ZOOM;
};

}

#endif
