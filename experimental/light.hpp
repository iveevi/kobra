#ifndef LIGHT_H_
#define LIGHT_H_

#include "object.hpp"

struct PointLight : public Object {
        // For now, just a position
        PointLight(const glm::vec3& position) : Object(position) {}
};

#endif
