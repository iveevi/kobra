#include "include/physics/rigidbody.hpp"

namespace mercury {

namespace physics {

// Constructors
RigidBody::RigidBody(float m, Transform *t, Collider *c)
	: CollisionBody(t, c), mass(m) {}

// Methods
void RigidBody::add_force(const glm::vec3 &force, float delta_t)
{
	// Semi-implicit euler integration
	velocity += delta_t * force/mass;
}

}

}