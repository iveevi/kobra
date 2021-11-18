#include "include/physics/rigidbody.hpp"

namespace mercury {

namespace physics {

void Rigidbody::add_force(const glm::vec3 &force, float delta_t)
{
	// Semi-implicit euler integration
	velocity += delta_t * force/mass;
}

}

}