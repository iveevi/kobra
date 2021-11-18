#ifndef PHYSICS_H_
#define PHYSICS_H_

// Standard headers
#include <vector>

// Engine headers
#include "include/physics/rigidbody.hpp"

namespace mercury {

namespace physics {

// Physics daemon
class Daemon {
        // List of rigid bodies
        std::vector <Rigidbody*> _rbs;
public:
        // Add a rigid body to the daemon
        void add_rb(Rigidbody *);

        // Run physics daemon
        void update(float);
};

}

}

#endif