#ifndef PHYSICS_H_
#define PHYSICS_H_

// Standard headers
#include <vector>

// Engine headers
#include "include/physics/collisionbody.hpp"
#include "include/physics/rigidbody.hpp"

namespace mercury {

namespace physics {

// Physics daemon
class Daemon {
        // List of rigid bodies
        std::vector <RigidBody *>	_rbs;

        // List of all objects with colliders
        std::vector <CollisionBody *>	_cbs;
public:
        // Adding collision bodies to the daemon
        void add_rb(RigidBody *);	// Must be called to add rigid bodies to the daemon
	void add_cb(CollisionBody *);

        // Run physics daemon
        void update(float);
};

}

}

#endif