#ifndef KOBRA_OPTIX_SBT_H_
#define KOBRA_OPTIX_SBT_H_

// Engine headers
#include "../bbox.hpp"
#include "../cuda/material.cuh"
#include "../vertex.hpp"

namespace kobra {

namespace optix {

// Hit data record
struct Hit {
	BoundingBox bbox;
	Vertex *vertices;
	glm::mat4 model;
	glm::uvec3 *triangles;
	int32_t light_index;
	uint32_t material_index;
};


}

}

#endif
