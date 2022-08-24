#ifndef KOBRA_CUDA_MATERIAL_H_
#define KOBRA_CUDA_MATERIAL_H_

// Engine headers
#include "math.cuh"
#include "../types.hpp"

namespace kobra {

namespace cuda {

struct Material {
	float3		diffuse;
	float3		specular;
	float3		emission;
	float3		ambient;
	float		shininess;
	float		roughness;
	float		refraction;
	Shading		type;
};

}

}

#endif
