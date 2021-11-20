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
                                rbf += c.mtv;
                                inter = true;
                                break;
                        }
                }

                // TODO: put in another function
                // Logger::notify() << "intersects? " << std::boolalpha << inter << "\n";
                if (!inter) {
                        rb->add_force(1000.0f * rbf, delta_t);
                        rb->add_force(rb->mass * gravity, delta_t);
                        rb->transform->move(rb->velocity * delta_t);
                }
        }
}

}

}