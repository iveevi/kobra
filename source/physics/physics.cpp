#include "include/physics/physics.hpp"

// TODO: remove
#include "include/mesh/sphere.hpp"
#include "include/ui/line.hpp"

namespace mercury {

namespace physics {

void Daemon::add_cobject(CollisionObject* co, float mass)
{
        _state.push_back(State {
                .mass = mass,
                .inv_mass = 1.0f / mass,
                .intertia = 2.0f,               // TODO: set default inside the daemon
                .inv_inertia = 0.5f,
                .skip = false,
                .co = co
        });
}

glm::quat deltaRotation(const glm::vec3 &em, float dt)
{
        glm::vec3 ha = em * dt * 0.5f; // vector of half angle
        float l = glm::length(ha);              // magnitude
        if (l > 0) {
                ha *= sin(l) / l;
                return glm::quat(cos(l), ha.x, ha.y, ha.z);
        }

        return glm::quat(1.0, ha.x, ha.y, ha.z);
}

void draw_point(const glm::vec3 &pt, const glm::vec3 &color, rendering::Daemon *rdam, Shader *shader)
{
        SVA3 *sva = new SVA3(mesh::wireframe_sphere(pt, 0.1f));
        sva->color = color;
        rdam->add(sva, shader);
}

void draw_line(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &color, rendering::Daemon *rdam, Shader *shader)
{
        ui::Line *line = new ui::Line(a, b);
        line->color = color;
        rdam->add(line, shader);
}

void Daemon::update(float delta_t, rendering::Daemon *rdam, Shader *shader)
{
        static const glm::vec3 gravity{0.0f, -9.81f, 0.0f};
        static const float dt = 1.0f / 1000.0f;                   // Fixed timestep: TODO: this is not enough...

        // TODO: repeat the physics while there is delta_t left

        // Drawing lambdas
        auto point = [&](const glm::vec3 &pt, const glm::vec3 &color) {
                draw_point(pt, color, rdam, shader);
        };

        auto line = [&](const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &color) {
                draw_line(a, b, color, rdam, shader);
        };

        // TODO: setup a state vector and matrix
        for (size_t i = 0; i < _state.size(); i++) {
                State &s = _state[i];

                if (s.skip)
                        continue;

                // To avoid duplicate handling, only check for
                // collisions past the current index
                for (size_t j = i + 1; j < _state.size(); j++) {
                        State &t = _state[j];
                        
                        Collision c = intersects(s.co->collider, t.co->collider, rdam, shader);

                        // Skip if no collision or the mtv is too small
                        if (!c.colliding || glm::length(c.mtv) < 1e-10f)
                                continue;
                        
                        // Draw contact_a
                        point(c.contact_a, {1.0f, 0.0f, 0.0f});
                        
                        // TODO: deal with both sides of the collision
                        if (s.co->type == CollisionObject::Type::DYNAMIC) {
                                // Get the object out of collision
                                s.co->transform->move(-c.mtv);

                                // Calculate the impulse
                                glm::vec3 P = 1.6f * s.mass * s.p * glm::normalize(c.mtv);

                                // s.p = -glm::length(s.p) * glm::normalize(c.mtv);
                                s.p += P;

                                glm::vec3 da = c.contact_a - s.co->transform->translation;
                                Logger::warn() << "da = " << da.x << ", " << da.y << ", " << da.z << std::endl;
                                Logger::warn() << "P = " << P.x << ", " << P.y << ", " << P.z << std::endl;
                                Logger::warn() << "mtv = " << c.mtv.x << ", " << c.mtv.y << ", " << c.mtv.z << std::endl;
                                Logger::warn() << "\tlength = " << glm::length(c.mtv) << std::endl;
                                Logger::warn() << "\tnormalized = " << glm::normalize(c.mtv).x << ", " << glm::normalize(c.mtv).y << ", " << glm::normalize(c.mtv).z << std::endl;
                                
                                glm::vec3 l = glm::cross(da, P);
                                s.l += l;

                                // s.p = glm::length(s.p) * glm::normalize(c.mtv);

                                // Assume that radius is 1.0f
                                // s.L += glm::vec3 {0.001, 0, 0.001};

                                glm::vec3 p = s.co->collider->support(c.mtv);
                                glm::vec3 o = s.co->transform->translation;

                                line(c.contact_a, c.contact_a + P, {1.0f, 0.0f, 0.0f});
                                line(c.contact_a, c.contact_a + 5.0f * c.mtv, {0.0f, 1.0f, 0.0f});
                                line(o, o + l, {0.0f, 1.0f, 0.0f});

                                point(p, {0.0f, 0.0f, 1.0f});
                                point(s.co->transform->translation, {0.0f, 1.0f, 1.0f});
                        }

                        // Wait for input
                        // std::cin.get();
                        s.skip = true;
                }

                // Add other forces
                if (s.co->type == CollisionObject::Type::DYNAMIC) {
                        s.p += s.mass * gravity * dt;
                }
        }

        // Apply momentums
        // TODO: some way to turn logs on and off
        Logger::warn() << "Moving state objects:\n";
        for (size_t i = 0; i < _state.size(); i++) {
                State &s = _state[i];

                // Skip if the object is static
                if (s.co->type == CollisionObject::Type::STATIC || s.skip)
                        continue;

                if (std::isnan(s.p.x) | std::isnan(s.p.y) | std::isnan(s.p.z))
                        Logger::fatal_error("nan component");
                
                Logger::notify() << "\ti = " << i << "\n";
                Logger::error() << "\t\ts.p = " << s.p.x << ", " << s.p.y << ", " << s.p.z << "\n";
                Logger::error() << "\t\ts.v before = " << s.v.x << ", " << s.v.y << ", " << s.v.z << "\n";

                s.v = s.p * s.inv_mass;
                s.w = s.l * s.inv_inertia;

                Logger::error() << "\t\ts.v after = " << s.v.x << ", " << s.v.y << ", " << s.v.z << "\n";

                // Calculate spin
                Logger::error() << "\ts.w = " << s.w.x << ", " << s.w.y << ", " << s.w.z << "\n";
                
                glm::quat q = 0.5f * glm::quat(s.w) * s.co->transform->orient * dt;
                // q *= glm::quat(s.w);
                // q *= 0.5f;
                // glm::quat spin = 0.5f * q * s.co->transform->orient;

                s.co->transform->move(s.v * dt);
                s.co->transform->rotate(deltaRotation(s.w, dt));
                // s.co->transform->orient += q;
                // s.co->transform->orient = glm::normalize(s.co->transform->orient);
        }
}

}

}