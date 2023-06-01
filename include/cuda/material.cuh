#ifndef KOBRA_CUDA_MATERIAL_H_
#define KOBRA_CUDA_MATERIAL_H_

// Engine headers
#include "math.cuh"
#include "../types.hpp"

namespace kobra {

namespace cuda {

// TODO: remove this...
struct Material {
	float3		diffuse;
	float3		specular;
	float3		emission;
	float3		ambient; // TODO: remove...
	float		shininess;
	float		roughness;
	float		refraction;
	Shading		type;
	int32_t		light_index;
};

// Uber material info for RTX
struct _material {
	float3		diffuse;
	float3		specular;
	float3		emission;
	float3		ambient;
	float		shininess;
	float		roughness;
	float		refraction;
	Shading		type;

	struct {
		cudaTextureObject_t	diffuse;
		cudaTextureObject_t	emission;
		cudaTextureObject_t	normal;
		cudaTextureObject_t	roughness;
		cudaTextureObject_t	specular;

		bool			has_diffuse = false;
		bool			has_emission = false;
		bool			has_normal = false;
		bool			has_roughness = false;
		bool			has_specular = false;
	} textures;
};

}

}

#endif
