#pragma once

// Engine headers
#include "include/system.hpp"

namespace kobra {

// TODO: template?
// then specialize comparison function and retrieval function?
struct TransformDaemon {
        enum : uint8_t {
                eSame,
                eChanged,
                eDynamic // Assumed to be always moving
        };

        // System reference, along with flag array
        System *system;
        std::vector <uint8_t> status;
        std::vector <kobra::Transform> transforms;

        TransformDaemon(System *ref) : system(ref) {}

        size_t size() const {
                return status.size();
        }

        // TODO: skip update function and just signal to update...
        void update() {
                int entities = system->size();
                // TODO: store by address, to avoid
                // issues when entities are removed...

                // If there are new entities, then initialize
                for (int i = 0; i < entities; i++) {
                        if (i < transforms.size()) {
                                // Already initialized
                                const kobra::Transform &t = system->get <kobra::Transform> (i);
                                kobra::Transform &old = transforms[i];

                                glm::vec3 dtr = t.position - old.position;
                                glm::vec3 drot = t.rotation - old.rotation;
                                glm::vec3 dsc = t.scale - old.scale;

                                if (glm::length(dtr) > 1e-3f || glm::length(drot) > 1e-3f || glm::length(dsc) > 1e-3f)
                                        status[i] = eChanged;
                                else
                                        status[i] = eSame;

                                old = t;
                        } else {
                                // Initialize with copy
                                transforms.push_back(system->get <kobra::Transform> (i));
                                status.push_back(eSame);
                        }
                }
        }

        uint8_t operator[](int index) const {
                // Returns the modification status
                return status[index];
        }

        uint8_t changed(int index) const {
                // Returns the modification status
                return status[index];
        }
};

}
