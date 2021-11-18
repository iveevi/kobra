#include "include/physics/physics.hpp"

namespace mercury {

namespace physics {

void Daemon::add_rb(Rigidbody* rb)
{
    _rbs.push_back(rb);
}

void Daemon::update(float delta_t)
{
        static const glm::vec3 gravity {0.0f, -9.81f, 0.0f};

        // TODO: check for collisions
        for (Rigidbody* rb : _rbs) {
                rb->add_force(gravity, delta_t);
                rb->transform->move(rb->velocity * delta_t);
        }
}

}

}