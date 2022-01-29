#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Standard headers
#include <iostream>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Transform class
struct Transform {
        // Basic properties
        glm::vec3 position;
        glm::quat rotation;
        glm::vec3 scale;

        // Directions
        glm::vec3 forward;
        glm::vec3 up;
        glm::vec3 right;

        // Constructors
        Transform();
        Transform(const glm::vec3 &pos) : position {pos},
                rotation {glm::quat()},
                scale {1.0, 1.0, 1.0},
                forward {0.0, 0.0, 1.0},
                up {0.0, 1.0, 0.0},
                right {1.0, 0.0, 0.0} {}
};

#endif
