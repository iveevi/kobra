#ifndef KOBRA_OPTIX_SBT_H_
#define KOBRA_OPTIX_SBT_H_

// Engine headers
#include "../bbox.hpp"
#include "../cuda/material.cuh"

namespace kobra {

namespace optix {

// Hit data record
struct Hit {
	// Mesh data
	float2			*texcoords;
	float3			*vertices;
	uint3			*triangles;

	float3			*normals;
	float3			*tangents;
	float3			*bitangents;

	// Auto UV mapping parameters
	BoundingBox		bbox;

	// Material and textures
	cuda::Material		material;

	struct {
		cudaTextureObject_t	diffuse;
		cudaTextureObject_t	normal;
		cudaTextureObject_t	roughness;

		bool			has_diffuse = false;
		bool			has_normal = false;
		bool			has_roughness = false;
	} textures;
};


}

}

#endif
