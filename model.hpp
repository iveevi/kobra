#ifndef MODEL_H_
#define MODEL_H_

// GLM headers
#include "glm/glm/ext/vector_float3.hpp"

namespace mercury {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 texcoord;
};

}

#endif
