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
        for (Rigidbody *rb : _rbs) {
                // For now, check that rb is not colliding with anything
                // TODO: later partition the space and check only those
                bool inter = false;
                for (Rigidbody *other : _rbs) {
                        if (rb == other && intersects(rb->collider, other->collider)) {
                                inter = true;
                                break;
                        }
                }

                if (!inter) {
                        rb->add_force(rb->mass * gravity, delta_t);
                        rb->transform->move(rb->velocity * delta_t);
                }
        }
}

}

}