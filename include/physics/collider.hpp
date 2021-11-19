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
	using Vertices = std::vector <glm::vec3>;

	// Members
	Transform *transform;

	// Constructors
	Collider(Transform *);

	// Get the AABB
	// TODO: we really only need a support(direction) function
	// to find furthest points on each aligned axis
	virtual AABB aabb() const = 0;

	// Get a list of vertices
	virtual Vertices vertices() const = 0;

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
	virtual Vertices vertices() const override;
};

}

}

#endif