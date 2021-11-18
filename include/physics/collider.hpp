#ifndef COLLIDER_H_
#define COLLIDER_H_

// Standard headers
#include <utility>

// Engine headers
#include "include/transform.hpp"
#include "include/rendering.hpp"
#include "include/physics/aabb.hpp"

namespace mercury {

namespace physics {

// Collider abstract base class
struct Collider {
	// Aliases	
	using Interval = std::pair <float, float>;

	// Members
	Transform *transform;

	// Constructors
	Collider(Transform *);

	// Get the AABB
	virtual AABB aabb() const = 0;

	// TODO: interval across SAT axis
};

// Default intersects method
bool intersects(Collider *, Collider *);

// Standard colliders
struct BoxCollider : public Collider {
	// Members
	glm::vec3 center;		// NOTE: These are local coordinates
	glm::vec3 size;			// No need for an up vector, comes from transform

	// Constructor
	BoxCollider(const glm::vec3 &, Transform *);
	BoxCollider(const glm::vec3 &, const glm::vec3 &, Transform *);

	// Methods
	void annotate(rendering::Daemon &, Shader *) const;

	// Virtual overrides
	virtual AABB aabb() const override;
};

}

}

#endif