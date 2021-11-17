#ifndef CUBE_H_
#define CUBE_H_

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "include/varray.hpp"

namespace mercury {

namespace mesh {

SVA3 wireframe_cuboid(const glm::vec3 &, const glm::vec3 &,
                const glm::vec3 &, const glm::vec3 & = {1.0, 1.0, 1.0});

SVA3 wireframe_cuboid(const glm::vec3 &, const glm::vec3 &,
        const glm::vec3 &, const glm::vec3 &, const glm::vec3 &);

}

}

#endif