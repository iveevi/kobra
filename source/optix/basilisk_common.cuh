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
	__constant__ kobra::optix::BasiliskParameters parameters;
}

// TODO: launch parameter for ray depth
#define MAX_DEPTH 3

// Local constants
static const float eps = 1e-3f;

// Ray packet data
struct RayPacket {
	float3	value;

	float3	position;
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
KCUDA_HOST_DEVICE
float3 Ld_Environment(float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	static const float eps = 0.05f;

	float3 contr_nee {0.0f};
	float3 contr_brdf {0.0f};

	// Sample random direction
	seed = random3(seed);
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

	// NEE
	float R = 1000; // TODO: world radius...

	float3 f = brdf(mat, n, wi, wo, entering, mat.type) * abs(dot(n, wi));

	// TODO: how to decide ray type for this?
	float pdf_light = 1.0f / (4.0f * M_PI * R * R);
	float pdf_brdf = pdf(mat, n, wi, wo, entering, mat.type);

	bool occluded = is_occluded(x, wi, R);
	if (!occluded) {
		float weight = power(pdf_light, pdf_brdf);
		contr_nee += weight * f * Li/pdf_light;
	}

	// BRDF
	Shading out;

	f = eval(mat, n, wo, entering, wi, pdf_brdf, out, seed) * abs(dot(n, wi));
	if (length(f) < 1e-6f)
		return contr_nee;

	occluded = is_occluded(x, wi, R);
	if (occluded)
		return contr_nee;

	u = atan2(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
	v = asin(wi.y)/M_PI + 0.5f;

	sample = tex2D <float4> (parameters.envmap, u, v);
	Li = make_float3(sample);
	
	float weight = 1.0f;
	if (out & eTransmission) // TODO: why this?
		return contr_nee;

	weight = power(pdf_brdf, pdf_light);

	// TODO: shoot shadow ray up to R
	if (pdf_light > 1e-9 && pdf_brdf > 1e-9)
		contr_brdf += weight * f * Li/pdf_brdf;

	return contr_nee + contr_brdf;
}

// Trace ray into scene and get relevant information
__device__ float3 Ld(float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.triangle_count;

	// TODO: launch parameter to control single light-single sample or
	// multiple light-single sample

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

	// if (quad_count == 0 && tri_count == 0)
	//	return make_float3(0.0f);

	// TODO: +1 for envmaps; make more efficient
	int total_count = quad_count + tri_count;

	unsigned int i = fract(random3(seed)).x * total_count;

	if (i < quad_count) {
		QuadLight light = parameters.lights.quads[i];
		return total_count * Ld_light(light, x, wo, n, mat, entering, seed);
	} else if (i < quad_count + tri_count) {
		int ni = i - quad_count;
		TriangleLight light = parameters.lights.triangles[ni];
		return total_count * Ld_light(light, x, wo, n, mat, entering, seed);
	} else {
		// Environment light
		return Ld_Environment(x, wo, n, mat, entering, seed);
	}

#endif

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
	if (rp->depth > MAX_DEPTH)					\
		return;

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
	if (isnan(n.x) || isnan(n.y) || isnan(n.z))			\
		return;

#endif
