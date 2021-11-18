#ifndef RIGIDBODY_H_
#define RIGIDBODY_H_

// Engine headers
#include "include/transform.hpp"
#include "include/physics/collider.hpp"

namespace mercury {

namespace physics {

// Rigidbody structure
struct Rigidbody {
	// Required sub-structures
	Transform *	transform;
	Collider *	collider;

	// Members
	glm::vec3	velocity;
	float		mass;
	// TODO: account for anglular velocity

	// NOTE: these functions do not account for collisions
	// only the physics daemon will apply the velocities and check for collisions
	void add_force(const glm::vec3 &, float);
	void add_torque(const glm::vec3 &, float);
};

}

}

#endif