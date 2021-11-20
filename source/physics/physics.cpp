#include "include/physics/physics.hpp"

namespace mercury {

namespace physics {

void Daemon::add_rb(RigidBody* rb)
{
        _rbs.push_back(rb);
        _cbs.push_back(rb);
}

void Daemon::add_cb(CollisionBody* cb)
{
        _cbs.push_back(cb);
}

void Daemon::update(float delta_t)
{
        static const glm::vec3 gravity{0.0f, -9.81f, 0.0f};
        static const float dt = 1.0f / 600.0f;                   // Fixed timestep

        // TODO: repeat the physics while there is delta_t left

        // TODO: check for collisions
        for (RigidBody *rb : _rbs) {    // Only loop through rigid bodies
                // For now, check that rb is not colliding with anything
                // TODO: later partition the space and check only those
                // TODO: need to check if the collision body is active (default yes)
                glm::vec3 rbf {0, 0, 0};
                
                bool inter = false;
                for (CollisionBody *other : _cbs) {
                        if (rb == other)
                                continue;

                        // TODO: maybe cache the collision results between colliders?
                        // then from the user's perpective, becomes simpler
                        Collision c = intersects(rb->collider, other->collider);
                        if (c.colliding) {
                                rbf -= glm::normalize(c.mtv);
                                inter = true;
                                break;
                        }
                }

                // TODO: put in another function
                if (!inter) {
                        rb->add_force(rb->mass * gravity, dt);
                } else {
                        // Kill velocity for now
                        rb->velocity = -0.8f * rb->velocity; //{0, 0, 0};
                }

                // Apply all forces        
                rb->transform->move(rb->velocity * dt);    // TODO: just have an rb method
        }
}

}

}