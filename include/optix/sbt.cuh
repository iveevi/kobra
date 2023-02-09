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
	glm::mat4 model;
	Vertex *vertices;
	glm::uvec3 *triangles;
	BoundingBox bbox;
	uint32_t material_index;
};


}

}

#endif
