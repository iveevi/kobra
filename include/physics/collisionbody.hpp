#ifndef COLLISION_BODY_H_
#define COLLISION_BODY_H_

// Engine headers
#include "include/transform.hpp"
#include "include/physics/collider.hpp"

namespace mercury {

namespace physics {

// NOTE: By default this acts like a Static Body
struct CollisionBody{
	// Required sub-structures
	Transform *	transform;	// TODO: copies pointer from collider
	Collider *	collider;

        // Constructors
        CollisionBody(Transform *, Collider *);
};

}

}

#endif