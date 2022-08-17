#ifndef KOBRA_LAYERS_OPTIX_TRACER_COMMON_H_
#define KOBRA_LAYERS_OPTIX_TRACER_COMMON_H_

// OptiX headers
#include <optix.h>

// Engine headers
#include "../cuda/math.cuh"

namespace kobra {

namespace optix_rt {

struct Params
{
	uchar4			*image;
	unsigned int		image_width;
	unsigned int		image_height;
	float3			cam_eye;
	float3			cam_u, cam_v, cam_w;
	OptixTraversableHandle	handle;
};

// Material type
struct Material {
	float3		diffuse;
	float3		specular;
	float3		emission;
	float3		ambient;
	float		shininess;
	float		roughness;
	float		refraction;
};

// Light type
struct AreaLight {
	float3 a;
	float3 ab;
	float3 ac;
	float3 intensity;

	__forceinline__ __device__ float area() {
		return length(cross(ab, ac));
	}
};

struct RayGenData
{
	// No data needed
};

struct MissData
{
	// Background color
	// TODO: background texture
	float3			bg_color;

	cudaTextureObject_t	bg_tex;
};

struct HitGroupData
{
	// Mesh data
	float2			*texcoords;
	float3			*vertices;
	uint3			*triangles;

	float3			*normals;
	float3			*tangents;
	float3			*bitangents;

	// Material and textures
	Material		material;

	struct {
		cudaTextureObject_t	diffuse;
		cudaTextureObject_t	normal;

		bool			has_diffuse;
		bool			has_normal;
	} textures;

	// Light data
	AreaLight		*area_lights;
	int			n_area_lights;
};

}

}

#endif
