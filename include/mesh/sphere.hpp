#ifndef SPHERE_H_
#define SPHERE_H_

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "include/varray.hpp"

namespace mercury {

namespace mesh {

// TODO: sphere mesh generator (icosphere, normal sphere, etc)
SVA3 wireframe_sphere(const glm::vec3 &, float, int = 64);

}

}

#endif