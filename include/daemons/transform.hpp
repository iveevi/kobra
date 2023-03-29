#pragma once

// Engine headers
#include "../ecs.hpp"

namespace kobra {

namespace daemons {

// TODO: template?
// then specialize comparison function and retrieval function?
struct Transform {
        enum : uint8_t {
                eSame,
                eChanged,
                eDynamic // Assumed to be always moving
        };

        // ECS reference, along with flag array
        ECS *ecs;
        std::vector <uint8_t> status;
        std::vector <kobra::Transform> transforms;

        Transform(ECS *ref) : ecs(ref) {}

        size_t size() const {
                return status.size();
        }

        void update() {
                int entities = ecs->size();
                // TODO: store by address, to avoid
                // issues when entities are removed...

                // If there are new entities, then initialize
                for (int i = 0; i < entities; i++) {
                        if (i < transforms.size()) {
                                // Already initialized
                                const kobra::Transform &t = ecs->get <kobra::Transform> (i);
                                kobra::Transform &old = transforms[i];

                                glm::vec3 dtr = t.position - old.position;
                                glm::vec3 drot = t.rotation - old.rotation;
                                glm::vec3 dsc = t.scale - old.scale;

                                if (glm::length(dtr) > 1e-3f || glm::length(drot) > 1e-3f || glm::length(dsc) > 1e-3f) {
                                        status[i] = eChanged;
                                        std::cout << "Entity index " << i << " has moved/transformed\n";
                                } else
                                        status[i] = eSame;

                                old = t;
                        } else {
                                // Initialize with copy
                                transforms.push_back(ecs->get <kobra::Transform> (i));
                                status.push_back(eSame);
                        }
                }
        }

        uint8_t operator[](int index) const {
                // Returns the modification status
                return status[index];
        }
};

}

}
