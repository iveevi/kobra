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

using namespace kobra;
using namespace kobra::cuda;
using namespace kobra::optix;

extern "C"
{
	__constant__ kobra::optix::BasiliskParameters parameters;
}

// TODO: launch parameter for ray depth
// TODO: rename to MAX_BOUNCES
#define MAX_DEPTH 3

// Local constants
static const float eps = 1e-3f;

// Ray packet data
struct RayPacket {
	float3	value;

	float4	position;
	float3	normal;
	float3	albedo;

	float	pdf;

	float3	wi;
	int	miss_depth;

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
		0, 0, 1,
		j0, j1
	);

	return vis;
}

// Get direct lighting for environment map
KCUDA_DEVICE
float3 Ld_Environment(const SurfaceHit &sh, float &pdf, Seed seed)
{
	static const float WORLD_RADIUS = 10000.0f;

	// Sample random direction
	seed = rand_uniform_3f(seed);
	float theta = acosf(sqrtf(1.0f - fract(seed.x)));
	float phi = 2.0f * M_PI * fract(seed.y);

	float3 wi = make_float3(
		sinf(theta) * cosf(phi),
		sinf(theta) * sinf(phi),
		cosf(theta)
	);

	float u = atan2(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(wi.y)/M_PI + 0.5f;

	float4 sample = tex2D <float4> (parameters.envmap, u, v);
	float3 Li = make_float3(sample);

	pdf = 1.0f / (4.0f * M_PI * WORLD_RADIUS * WORLD_RADIUS);

	// NEE
	bool occluded = is_occluded(sh.x, wi, WORLD_RADIUS);
	if (occluded)
		return make_float3(0.0f);

	// TODO: method for abs dot in surface hit
	float3 f = brdf(sh, wi, eDiffuse) * abs(dot(sh.n, wi));

	// TODO: how to decide ray type for this?
	return f * Li;
}

// Trace ray into scene and get relevant information
__device__
float3 Ld(const SurfaceHit &sh, Seed seed)
{
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.triangle_count;

	// TODO: parameter for if envmap is used
	int total_count = quad_count + tri_count + parameters.has_envmap;

	// Regular direct lighting
	unsigned int i = rand_uniform(seed) * total_count;

	float3 contr = {0.0f};
	float pdf = 1.0f/total_count;

	float light_pdf = 0.0f;
	if (i < quad_count) {
		QuadLight light = parameters.lights.quads[i];
		contr = Ld_light(light, sh, light_pdf, seed);
	} else if (i < quad_count + tri_count) {
		int ni = i - quad_count;
		TriangleLight light = parameters.lights.triangles[ni];
		contr = Ld_light(light, sh, light_pdf, seed);
	} else {
		// Environment light
		// TODO: imlpement PBRT's better importance sampling
		contr = Ld_Environment(sh, light_pdf, seed);
	}

	pdf *= light_pdf;
	return contr/pdf;
}

// Uniformly sample a single light source
struct FullLightSample {
	// Sample information
	float3 Le;
	float3 normal;
	float3 point;
	float pdf;

	// Light information
	int type; // 0 - quad, 1 - triangle
	int index;
};

KCUDA_INLINE KCUDA_DEVICE
FullLightSample sample_direct(Seed seed)
{
	// TODO: plus envmap
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.triangle_count;

	unsigned int total_lights = quad_count + tri_count + parameters.has_envmap;
	unsigned int light_index = rand_uniform(seed) * total_lights;

	FullLightSample sample;
	if (light_index < quad_count) {
		// Get quad light
		QuadLight light = parameters.lights.quads[light_index];

		// Sample point
		float3 point = sample_area_light(light, seed);

		// Copy information
		sample.Le = light.intensity;
		sample.normal = light.normal();
		sample.point = point;
		sample.pdf = 1.0f/(light.area() * total_lights);

		sample.type = 0;
		sample.index = light_index;
	} else if (light_index < quad_count + tri_count) {
		// Get triangle light
		int ni = light_index - quad_count;
		TriangleLight light = parameters.lights.triangles[ni];

		// Sample point
		float3 point = sample_area_light(light, seed);

		// Copy information
		sample.Le = light.intensity;
		sample.normal = light.normal();
		sample.point = point;
		sample.pdf = 1.0f/(light.area() * total_lights);

		sample.type = 1;
		sample.index = ni;
	} else {
		// Sample environment light
		seed = rand_uniform_3f(seed);

		float theta = acosf(sqrtf(1.0f - fract(seed.x)));
		float phi = 2.0f * M_PI * fract(seed.y);

		float3 wi = make_float3(
			sinf(theta) * cosf(phi),
			sinf(theta) * sinf(phi),
			cosf(theta)
		);

		// TODO: world radius in parameters
		float3 point = wi * 10000.0f;

		float u = atan2(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
		float v = asin(wi.y)/M_PI + 0.5f;

		float4 env = tex2D <float4> (parameters.envmap, u, v);

		float pdf = 1.0f/(4.0f * M_PI * 1000.0f * 1000.0f * total_lights);
		
		// Copy information
		sample.Le = make_float3(env);
		sample.normal = make_float3(0.0f);
		sample.point = point;
		sample.pdf = pdf;

		sample.type = 2;
		sample.index = -1;
	}

	return sample;
}

// Compute direct lighting for a given sample
__device__ __forceinline__
float3 direct_unoccluded(const SurfaceHit &sh, float3 Le, float3 normal, int type, float3 D, float d)
{
	// Assume that the light is visible
	// TODO: evaluate all lobes...
	float3 rho = cuda::brdf(sh, D, eDiffuse);

	float geometric = abs(dot(sh.n, D))/(d * d);

	// TODO: enums
	if (type != 2) {
		float ldot = abs(dot(normal, D));
		geometric *= ldot;
	}

	return rho * Le * geometric;
}

__device__ __forceinline__
float3 direct_occluded(const SurfaceHit &sh, float3 Le, float3 normal, int type, float3 D, float d)
{
	bool occluded = is_occluded(sh.x, D, d);

	float3 Li = make_float3(0.0f);
	if (!occluded) {
		float3 rho = brdf(sh, D, eDiffuse);

		float geometric = abs(dot(sh.n, D))/(d * d);
		if (type != 2) {
			float ldot = abs(dot(normal, D));
			geometric *= ldot;
		}

		Li = rho * Le * geometric;
	}

	return Li;
}

// Updating a reservoir
__device__ __forceinline__
bool reservoir_update(LightReservoir *reservoir, const LightSample &sample, float weight, Seed seed)
{
	reservoir->weight += weight;

	float eta = rand_uniform(seed);
	bool selected = (eta * reservoir->weight) < weight;

	if (selected)
		reservoir->sample = sample;

	reservoir->count++;

	return selected;
}

// Kernel helpers/code blocks
template <unsigned int Mode>
KCUDA_INLINE __device__
void trace(float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(parameters.traversable,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		Mode, eCount, 0,
		i0, i1
	);
}

#define LOAD_RAYPACKET()						\
	RayPacket *rp;							\
	unsigned int i0 = optixGetPayload_0();				\
	unsigned int i1 = optixGetPayload_1();				\
	rp = unpack_pointer <RayPacket> (i0, i1);			\
	if (rp->depth > MAX_DEPTH) {					\
		rp->value = make_float3(0.0f);				\
		return;							\
	}

#define LOAD_INTERSECTION_DATA()					\
	Hit *hit = reinterpret_cast <Hit *>				\
		(optixGetSbtDataPointer());				\
									\
	float2 bary = optixGetTriangleBarycentrics();			\
	int primitive_index = optixGetPrimitiveIndex();			\
	uint3 triangle = hit->triangles[primitive_index];		\
									\
	float2 uv = interpolate(hit->texcoords, triangle, bary);	\
	uv.y = 1 - uv.y;						\
									\
	Material material = hit->material;				\
	calculate_material(hit, material, triangle, uv);		\
									\
	bool entering;							\
	float3 wo = -optixGetWorldRayDirection();			\
	float3 n = calculate_normal(hit, triangle, bary, uv, entering);	\
	float3 x = interpolate(hit->vertices, triangle, bary);		\
									\
	if (isnan(n.x) || isnan(n.y) || isnan(n.z)) {			\
		rp->value = make_float3(0.0f);				\
		return;							\
	}

#endif
