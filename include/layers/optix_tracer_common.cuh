#ifndef KOBRA_LAYERS_OPTIX_TRACER_COMMON_H_
#define KOBRA_LAYERS_OPTIX_TRACER_COMMON_H_

// OptiX headers
#include <optix.h>

// Engine headers
#include "../cuda/math.cuh"
#include "../types.hpp"

namespace kobra {

namespace optix_rt {

struct Params
{
	float3			*pbuffer;

	uchar4			*image;
	unsigned int		image_width;
	unsigned int		image_height;

	float3			cam_eye;
	float3			cam_u;
	float3			cam_v;
	float3			cam_w;

	float			time;

	int			accumulated;

	OptixTraversableHandle	handle;
	OptixTraversableHandle	handle_shadow;
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
	Shading		type;
};

__forceinline__ __device__ bool intersects_triangle
		(float3 v1, float3 v2, float3 v3,
		 float3 origin, float3 dir)
{
	float3 e1 = v2 - v1;
	float3 e2 = v3 - v1;
	float3 s1 = cross(dir, e2);
	float divisor = dot(s1, e1);
	if (divisor == 0.0)
		return false;
	float3 s = origin - v1;
	float inv_divisor = 1.0 / divisor;
	float b1 = dot(s, s1) * inv_divisor;
	if (b1 < 0.0 || b1 > 1.0)
		return false;
	float3 s2 = cross(s, e1);
	float b2 = dot(dir, s2) * inv_divisor;
	if (b2 < 0.0 || b1 + b2 > 1.0)
		return false;
	return true;
}

// Light type
struct AreaLight {
	float3 a;
	float3 ab;
	float3 ac;
	float3 intensity;

	__forceinline__ __device__ float area() {
		return length(cross(ab, ac));
	}

	__forceinline__ __device__ float3 normal() {
		return normalize(cross(ab, ac));
	}

	__forceinline__ __device__ bool intersects(float3 origin, float3 dir) {
		float3 v1 = a;
		float3 v2 = a + ab;
		float3 v3 = a + ac;
		float3 v4 = a + ab + ac;

		return intersects_triangle(v1, v2, v3, origin, dir) ||
			   intersects_triangle(v2, v3, v4, origin, dir);
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
