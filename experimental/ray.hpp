#ifndef RAY_H_
#define RAY_H_

// GLM headers
#include <glm/glm.hpp>

// Ray object
struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;

        // Calculate vector at time
        glm::vec3 at(float t) const {
                return origin + t * direction;
        }
};

#endif
