#ifndef PHYSICS_H_
#define PHYSICS_H_

// Standard headers
#include <vector>

// Engine headers
#include "include/physics/collision_object.hpp"
// #include "include/physics/rigidbody.hpp"

namespace mercury {

namespace physics {

// Physics daemon
// TODO: add an rdam as a member variable for optional annotations
class Daemon {
        // State object
        struct State {
                // Physical properties
                float mass;
                float inv_mass;

                float intertia;
                float inv_inertia;

                // Dynamics attributes
                glm::vec3 v;            // Linear velocity
                glm::vec3 p;            // Linear momentum

                glm::vec3 w;            // Angular velocity
                glm::vec3 l;            // Angular momentum

                // Skip physics
                bool skip;

                // Collision object
                CollisionObject* co;
        };

        // List of all objects with colliders
        std::vector <State>	_state;
public:
        // Adding collision bodies to the daemon
	void add_cobject(CollisionObject *, float);	// TODO: add other properties (as a struct)

        // TODO: account for different collision algorithms
        // TODO: account for different integration methods (euler, verlet, etc.)

        // Run physics daemon
        void update(float, rendering::Daemon *, Shader *);
};

}

}

#endif