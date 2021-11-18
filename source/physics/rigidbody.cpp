#include "include/physics/rigidbody.hpp"

namespace mercury {

namespace physics {

// Constructors
Rigidbody::Rigidbody(float m, Transform *t, Collider *c)
	: mass(m), transform(t), collider(c) {}

// Methods
void Rigidbody::add_force(const glm::vec3 &force, float delta_t)
{
	// Semi-implicit euler integration
	velocity += delta_t * force/mass;
}

}

}