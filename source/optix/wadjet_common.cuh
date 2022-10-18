#ifndef KOBRA_OPTIX_WADJET_COMMON_H_
#define KOBRA_OPTIX_WADJET_COMMON_H_

// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/cuda/brdf.cuh"
#include "../../include/cuda/material.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/optix/parameters.cuh"
#include "../../include/optix/lighting.cuh"
#include "../../include/cuda/matrix.cuh"

using namespace kobra::cuda;
using namespace kobra::optix;

extern "C"
{
	__constant__ kobra::optix::WadjetParameters parameters;
}

// TODO: launch parameter for ray depth
// #define MAX_DEPTH 10
#define MAX_DEPTH 3

// Local constants
static const float eps = 1e-3f;

// Ray packet data
struct RayPacket {
	float3	value;
	float3	position;
	float3	normal;

	float	ior;
	
	int	depth;
	uint	index;
	
	float3	seed;
};

// Interpolate triangle values
template <class T>
KCUDA_INLINE __device__
T interpolate(T *arr, uint3 triagle, float2 bary)
{
	T a = arr[triagle.x];
	T b = arr[triagle.y];
	T c = arr[triagle.z];

	return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

// Calculate hit normal
static __device__ float3 calculate_normal
		(Hit *hit_data, uint3 triangle,
		 float2 bary, float2 uv, bool &entering)
{
	float3 e1 = hit_data->vertices[triangle.y] - hit_data->vertices[triangle.x];
	float3 e2 = hit_data->vertices[triangle.z] - hit_data->vertices[triangle.x];
	float3 ng = cross(e1, e2);

	if (dot(ng, optixGetWorldRayDirection()) > 0.0f) {
		ng = -ng;
		entering = false;
	} else {
		entering = true;
	}

	ng = normalize(ng);

	float3 normal = interpolate(hit_data->normals, triangle, bary);
	if (dot(normal, ng) < 0.0f)
		normal = -normal;

	normal = normalize(normal);

	if (hit_data->textures.has_normal) {
		float4 n4 = tex2D <float4> (hit_data->textures.normal, uv.x, uv.y);
		float3 n = 2 * make_float3(n4.x, n4.y, n4.z) - 1;

		// Tangent and bitangent
		float3 tangent = interpolate(hit_data->tangents, triangle, bary);
		float3 bitangent = interpolate(hit_data->bitangents, triangle, bary);

		mat3 tbn = mat3(
			normalize(tangent),
			normalize(bitangent),
			normalize(normal)
		);

		normal = normalize(tbn * n);
	}

	return normal;
}

// Calculate relevant material data for a hit
KCUDA_INLINE __device__
void calculate_material(Hit *hit_data, Material &mat, uint3 triangle, float2 uv)
{
	if (hit_data->textures.has_diffuse) {
		float4 d4 = tex2D <float4> (hit_data->textures.diffuse, uv.x, uv.y);
		mat.diffuse = make_float3(d4);
	}

	if (hit_data->textures.has_roughness) {
		float4 r4 = tex2D <float4> (hit_data->textures.roughness, uv.x, uv.y);
		mat.roughness = r4.x;
	}
}

// Check shadow visibility
KCUDA_INLINE __device__
bool is_occluded(float3 origin, float3 dir, float R)
{
	static float eps = 0.05f;

	bool vis = true;

	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	optixTrace(parameters.traversable,
		origin, dir,
		0, R - eps, 0,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
			| OPTIX_RAY_FLAG_DISABLE_ANYHIT
			| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		2 * parameters.instances, 0, 1,
		j0, j1
	);

	return vis;
}

// Trace ray into scene and get relevant information
__device__ float3 Ld(float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.triangle_count;

// #define GROUND_TRUTH

#ifdef GROUND_TRUTH
	
	float3 contr = {0.0f};

	for (int i = 0; i < quad_count; i++) {
		QuadLight light = parameters.lights.quads[i];
		contr += Ld_light(light, x, wo, n, mat, entering, seed);
	}

	for (int i = 0; i < tri_count; i++) {
		TriangleLight light = parameters.lights.triangles[i];
		contr += Ld_light(light, x, wo, n, mat, entering, seed);
	}

	return contr;

#else

	if (quad_count == 0 && tri_count == 0)
		return make_float3(0.0f);

	int total_count = quad_count + tri_count;

	random3(seed);
	unsigned int i = fract(seed.x) * (quad_count + tri_count);
	i = min(i, quad_count + tri_count - 1);

	if (i < quad_count) {
		QuadLight light = parameters.lights.quads[i];
		return total_count * Ld_light(light, x, wo, n, mat, entering, seed);
	}

	TriangleLight light = parameters.lights.triangles[i - quad_count];
	return total_count * Ld_light(light, x, wo, n, mat, entering, seed);

#endif

}

// Trace specializations
__device__
void trace_regular(float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(parameters.traversable,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		WadjetParameters::eRegular, WadjetParameters::eCount, 0,
		i0, i1
	);
}

__device__
void trace_restir(float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(parameters.traversable,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		WadjetParameters::eReSTIR, WadjetParameters::eCount, 0,
		i0, i1
	);
}

__device__
void trace_voxel(float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(parameters.traversable,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		WadjetParameters::eVoxel, WadjetParameters::eCount, 0,
		i0, i1
	);
}

#endif
