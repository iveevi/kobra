#ifndef COLLISION_OBJECT_H_
#define COLLISION_OBJECT_H_

// Engine headers
#include "include/transform.hpp"
#include "include/physics/collider.hpp"

namespace mercury {

namespace physics {

// NOTE: By default this acts like a Static Body
struct CollisionObject {
	// Types of collision bodies
	enum class Type {
		STATIC,
		DYNAMIC,
		KINEMATIC
		// TODO: area collision body
	};

	// Required sub-structures
	Transform *	transform;	// TODO: copies pointer from collider
	Collider *	collider;
	Type		type;

	// Optional sub-structures
	// TODO: Add a handler for collision events

        // Constructors
        CollisionObject(Collider *, Type = Type::STATIC);
        CollisionObject(Transform *, Collider *, Type = Type::STATIC);
};

}

}

#endif