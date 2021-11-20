#include "include/physics/physics.hpp"

namespace mercury {

namespace physics {

void Daemon::add_cobject(CollisionObject* co, float mass)
{
        _state.push_back(State {
                .mass = mass,
                .inv_mass = 1.0f / mass,
                .co = co
        });
}

void Daemon::update(float delta_t)
{
        static const glm::vec3 gravity{0.0f, -9.81f, 0.0f};
        static const float dt = 1.0f / 600.0f;                   // Fixed timestep: TODO: this is not enough...

        // TODO: repeat the physics while there is delta_t left

        // TODO: setup a state vector and matrix
        for (size_t i = 0; i < _state.size(); i++) {
                State &s = _state[i];

                // To avoid duplicate handling, only check for
                // collisions past the current index
                for (size_t j = i + 1; j < _state.size(); j++) {
                        State &t = _state[j];
                        
                        Collision c = intersects(s.co->collider, t.co->collider);
                        if (!c.colliding)
                                continue;
                        
                        // TODO: deal with both sides of the collision
                        if (s.co->type == CollisionObject::Type::DYNAMIC) {
                                // s.co->transform->move(c.mtv);
                                // s.p = glm::length(s.p) * glm::normalize(c.mtv);
                                Logger::notify() << "Collision happened: mtv = " << c.mtv.x << ", " << c.mtv.y << ", " << c.mtv.z << std::endl;
                                Logger::notify() << "Position = " << s.co->transform->translation.x << ", " << s.co->transform->translation.y << ", " << s.co->transform->translation.z << std::endl;
                                std::cin.get();
                                s.p = -s.p;
                        }
                }

                // Add other forces
                if (s.co->type == CollisionObject::Type::DYNAMIC) {
                        s.p += s.mass * gravity * dt;
                }
        }

        // Apply momentums
        // Logger::warn() << "Moving state objects:\n";
        for (size_t i = 0; i < _state.size(); i++) {
                State &s = _state[i];
                
                Logger::notify() << "\ti = " << i << "\n";
                Logger::error() << "\t\ts.p = " << s.p.x << ", " << s.p.y << ", " << s.p.z << "\n";
                Logger::error() << "\t\ts.v before = " << s.v.x << ", " << s.v.y << ", " << s.v.z << "\n";

                s.v = s.p * s.inv_mass;

                Logger::error() << "\t\ts.v after = " << s.v.x << ", " << s.v.y << ", " << s.v.z << "\n";

                s.co->transform->move(s.v * dt);
        }
}

}

}