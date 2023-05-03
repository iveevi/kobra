#ifndef KOBRA_OPTIX_WADJET_COMMON_H_
#define KOBRA_OPTIX_WADJET_COMMON_H_

// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/cuda/brdf.cuh"
#include "../../include/cuda/material.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/matrix.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/optix/lighting.cuh"
#include "../../include/optix/sbt.cuh"

using namespace kobra::cuda;
using namespace kobra::optix;

// TODO: launch parameter for ray depth

// Local constants
static const float eps = 1e-3f;

// Generic lighting context
struct LightingContext {
	OptixTraversableHandle handle;

	QuadLight *quads;
	TriangleLight *triangles;
	int quad_count;
	int triangle_count;

	bool has_envmap;
	cudaTextureObject_t envmap;

	KCUDA_DEVICE
	LightingContext(OptixTraversableHandle _handle,
			QuadLight *_quads,
			TriangleLight *_triangles,
			int _quad_count,
			int _triangle_count,
			bool _has_envmap,
			cudaTextureObject_t _envmap) :
		handle(_handle),
		quads(_quads),
		triangles(_triangles),
		quad_count(_quad_count),
		triangle_count(_triangle_count),
		has_envmap(_has_envmap),
		envmap(_envmap) {}
};

// Interpolate triangle values
template <class T>
KCUDA_INLINE KCUDA_DEVICE
T interpolate(const T &a, const T &b, const T &c, float2 bary)
{
	return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

// Compute hit point
static KCUDA_INLINE KCUDA_DEVICE
float3 calculate_intersection(Hit *hit, glm::uvec3 triangle, float2 bary)
{
	glm::vec3 a = hit->vertices[triangle.x].position;
	glm::vec3 b = hit->vertices[triangle.y].position;
	glm::vec3 c = hit->vertices[triangle.z].position;
	glm::vec3 x = interpolate(a, b, c, bary);
	x = hit->model * glm::vec4(x, 1.0f);
	return { x.x, x.y, x.z };
}

// Calculate hit normal
static __device__ float3 calculate_normal
		(Hit *hit_data, const _material &mat, glm::uvec3 triangle,
		float2 bary, glm::vec2 uv, bool &entering)
{
	glm::vec3 a = hit_data->vertices[triangle.x].position;
	glm::vec3 b = hit_data->vertices[triangle.y].position;
	glm::vec3 c = hit_data->vertices[triangle.z].position;

	// TODO: compute cross, then transform?
	glm::vec3 e1 = b - a;
	glm::vec3 e2 = c - a;

	e1 = hit_data->model * glm::vec4(e1, 0.0f);
	e2 = hit_data->model * glm::vec4(e2, 0.0f);

	glm::vec3 gnormal = glm::normalize(glm::cross(e1, e2));

	float3 ng = { gnormal.x, gnormal.y, gnormal.z };
	if (dot(ng, optixGetWorldRayDirection()) > 0.0f) {
		ng = -ng;
		entering = false;
	} else {
		entering = true;
	}

	ng = normalize(ng);

	a = hit_data->vertices[triangle.x].normal;
	b = hit_data->vertices[triangle.y].normal;
	c = hit_data->vertices[triangle.z].normal;

	gnormal = interpolate(a, b, c, bary);
	gnormal = hit_data->model * glm::vec4(gnormal, 0.0f);

	float3 normal = { gnormal.x, gnormal.y, gnormal.z };
	if (dot(normal, ng) < 0.0f)
		normal = -normal;

	normal = normalize(normal);

	if (mat.textures.has_normal) {
		float4 n4 = tex2D <float4> (mat.textures.normal, uv.x, uv.y);
		float3 n = 2 * make_float3(n4.x, n4.y, n4.z) - 1;

		// Tangent and bitangent
		a = hit_data->vertices[triangle.x].tangent;
		b = hit_data->vertices[triangle.y].tangent;
		c = hit_data->vertices[triangle.z].tangent;

		glm::vec3 gtangent = interpolate(a, b, c, bary);
		gtangent = hit_data->model * glm::vec4(gtangent, 0.0f);

		a = hit_data->vertices[triangle.x].bitangent;
		b = hit_data->vertices[triangle.y].bitangent;
		c = hit_data->vertices[triangle.z].bitangent;

		glm::vec3 gbitangent = interpolate(a, b, c, bary);
		gbitangent = hit_data->model * glm::vec4(gbitangent, 0.0f);

		gtangent = glm::normalize(hit_data->model * glm::vec4(gtangent, 0.0f));
		gbitangent = glm::normalize(hit_data->model * glm::vec4(gbitangent, 0.0f));

		float3 tangent = { gtangent.x, gtangent.y, gtangent.z };
		float3 bitangent = { gbitangent.x, gbitangent.y, gbitangent.z };

		// TODO: get rid of this
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
Material calculate_material(const _material &mat, glm::vec2 uv)
{
	Material material;
	material.diffuse = mat.diffuse;
	material.specular = mat.specular;
	material.emission = mat.emission;
	material.ambient = mat.ambient;
	material.shininess = mat.shininess;
	material.roughness = mat.roughness;
	material.refraction = mat.refraction;
	material.type = mat.type;

	if (mat.textures.has_diffuse) {
		float4 d4 = tex2D <float4> (mat.textures.diffuse, uv.x, uv.y);
		material.diffuse = make_float3(d4);
	}

	if (mat.textures.has_specular) {
		float4 s4 = tex2D <float4> (mat.textures.specular, uv.x, uv.y);
		material.specular = make_float3(s4);
	}

	if (mat.textures.has_emission) {
		float4 e4 = tex2D <float4> (mat.textures.emission, uv.x, uv.y);
		material.emission = make_float3(e4);
	}

	if (mat.textures.has_roughness) {
		float4 r4 = tex2D <float4> (mat.textures.roughness, uv.x, uv.y);
		material.roughness = r4.x;
	}

	return material;
}

// Check shadow visibility
KCUDA_INLINE __device__
bool is_occluded(OptixTraversableHandle handle, float3 origin, float3 dir, float R)
{
	static float eps = 0.05f;

	bool vis = true;

	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	optixTrace(handle,
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
float3 Ld_Environment(const LightingContext &lc, const SurfaceHit &sh, float &pdf, Seed seed)
{
	// TODO: sample in UV space instead of direction...
	static const float WORLD_RADIUS = 10000.0f;

	// Sample random direction
	seed = rand_uniform_3f(seed);
	float theta = acosf(sqrtf(1.0f - fract(seed.x)));
	float phi = 2.0f * PI * fract(seed.y);

	float3 wi = make_float3(
		sinf(theta) * cosf(phi),
		sinf(theta) * sinf(phi),
		cosf(theta)
	);

	float u = atan2f(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
	float v = asinf(wi.y)/M_PI + 0.5f;

	float4 sample = tex2D <float4> (lc.envmap, u, v);
	float3 Li = make_float3(sample);

	// pdf = 1.0f / (4.0f * M_PI * WORLD_RADIUS * WORLD_RADIUS);

	pdf = 1.0f/(4.0f * M_PI);

	// NEE
	bool occluded = is_occluded(lc.handle, sh.x, wi, WORLD_RADIUS);
	if (occluded)
		return make_float3(0.0f);

	// TODO: method for abs dot in surface hit
	float3 f = brdf(sh, wi, eDiffuse) * abs(dot(sh.n, wi));

	// TODO: how to decide ray type for this?
	return f * Li;
}

// Trace ray into scene and get relevant information
__device__
float3 Ld(const LightingContext &lc, const SurfaceHit &sh, Seed seed)
{
	int quad_count = lc.quad_count;
	int tri_count = lc.triangle_count;

	// TODO: parameter for if envmap is used
	int total_count = quad_count + tri_count + lc.has_envmap;

	// Regular direct lighting
	unsigned int i = rand_uniform(seed) * total_count;

	float3 contr = {0.0f};
	float pdf = 1.0f/total_count;

	float light_pdf = 0.0f;
	if (i < quad_count) {
		QuadLight light = lc.quads[i];
		contr = Ld_light(lc.handle, light, sh, light_pdf, seed);
	} else if (i < quad_count + tri_count) {
		int ni = i - quad_count;
		TriangleLight light = lc.triangles[ni];
		contr = Ld_light(lc.handle, light, sh, light_pdf, seed);
	} else {
		// Environment light
		// TODO: imlpement PBRT's better importance sampling
		contr = Ld_Environment(lc, sh, light_pdf, seed);
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
FullLightSample sample_direct(const LightingContext &lc, const SurfaceHit &sh, Seed seed)
{
	// TODO: plus envmap
	int quad_count = lc.quad_count;
	int tri_count = lc.triangle_count;

	unsigned int total_lights = quad_count + tri_count + lc.has_envmap;
	unsigned int light_index = rand_uniform(seed) * total_lights;

	FullLightSample sample;
	if (light_index < quad_count) {
		// Get quad light
		QuadLight light = lc.quads[light_index];

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
		TriangleLight light = lc.triangles[ni];

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
		float3 point = sh.x + wi * 10000.0f;

		float u = atan2(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
		float v = asin(wi.y)/M_PI + 0.5f;

		float4 env = tex2D <float4> (lc.envmap, u, v);

		float pdf = 1.0f/(4.0f * M_PI * total_lights);

		// Copy information
		sample.Le = make_float3(env);
		sample.normal = -wi;
		sample.point = point;
		sample.pdf = pdf;

		sample.type = 2;
		sample.index = -1;
	}

	return sample;
}

// Compute direct lighting for a given sample
__device__ __forceinline__
float3 direct_unoccluded(const SurfaceHit &sh,
		float3 Le, float3 normal, int type, float3 D, float d)
{
	// Assume that the light is visible
	// TODO: evaluate all lobes...
	float3 rho = brdf(sh, D, eDiffuse);

	float geometric = abs(dot(sh.n, D));

	// TODO: enums
	if (type != 2) {
		float ldot = abs(dot(normal, D));
		geometric *= ldot/(d * d);
	}

	return rho * Le * geometric;
}

__device__ __forceinline__
float3 direct_occluded(OptixTraversableHandle handle,
		const SurfaceHit &sh,
		float3 Le,
		float3 normal, int type, float3 D, float d)
{
	bool occluded = is_occluded(handle, sh.x, D, d);

	float3 Li = make_float3(0.0f);
	if (!occluded) {
		float3 rho = brdf(sh, D, eDiffuse);

		float geometric = abs(dot(sh.n, D));
		if (type != 2) {
			float ldot = abs(dot(normal, D));
			geometric *= ldot/(d * d);
		}

		Li = rho * Le * geometric;
	}

	return Li;
}

// Using FullLightSample
__device__ __forceinline__
float3 direct_unoccluded(const SurfaceHit &sh, const FullLightSample &fls, float3 D, float d)
{
	// Assume that the light is visible
	// TODO: evaluate all lobes...
	float3 rho = brdf(sh, D, eDiffuse);

	float geometric = abs(dot(sh.n, D));

	// TODO: enums
	if (fls.type != 2) {
		float ldot = abs(dot(fls.normal, D));
		geometric *= ldot/(d * d);
	}

	return rho * fls.Le * geometric;
}

__device__ __forceinline__
float3 direct_occluded(OptixTraversableHandle handle,
		const SurfaceHit &sh,
		const FullLightSample &fls,
		float3 D, float d)
{
	bool occluded = is_occluded(handle, sh.x, D, d);

	float3 Li = make_float3(0.0f);
	if (!occluded) {
		float3 rho = brdf(sh, D, eDiffuse);

		float geometric = abs(dot(sh.n, D));
		if (fls.type != 2) {
			float ldot = abs(dot(fls.normal, D));
			geometric *= ldot/(d * d);
		}

		Li = rho * fls.Le * geometric;
	}

	return Li;
}

// Kernel helpers/code blocks
KCUDA_INLINE __device__
void trace(OptixTraversableHandle handle, int hit_program, int stride, float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(handle,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		hit_program, stride, 0,
		i0, i1
	);
}

#define LOAD_RAYPACKET(parameters)						\
	RayPacket *rp;								\
	unsigned int i0 = optixGetPayload_0();					\
	unsigned int i1 = optixGetPayload_1();					\
	rp = unpack_pointer <RayPacket> (i0, i1);				\
	if (rp->depth > parameters.max_depth) {					\
		rp->value = make_float3(0.0f);					\
		return;								\
	}

#define LOAD_INTERSECTION_DATA(parameters)					\
	Hit *hit = reinterpret_cast <Hit *>					\
		(optixGetSbtDataPointer());					\
										\
	float2 bary = optixGetTriangleBarycentrics();				\
	int primitive_index = optixGetPrimitiveIndex();				\
	glm::uvec3 triangle = hit->triangles[primitive_index];			\
										\
	glm::vec2 uv_a = hit->vertices[triangle.x].tex_coords;			\
	glm::vec2 uv_b = hit->vertices[triangle.y].tex_coords;			\
	glm::vec2 uv_c = hit->vertices[triangle.z].tex_coords;			\
	glm::vec2 uv = interpolate(uv_a, uv_b, uv_c, bary);			\
	uv.y = 1 - uv.y;							\
										\
	_material mat = parameters.materials[hit->material_index];		\
	Material material = calculate_material(mat, uv);			\
										\
	bool entering;								\
	float3 wo = -optixGetWorldRayDirection();				\
	float3 n = calculate_normal(hit, mat, triangle, bary, uv, entering);	\
	float3 x = calculate_intersection(hit, triangle, bary);			\
										\
	if (isnan(n.x) || isnan(n.y) || isnan(n.z)) {				\
		rp->value = make_float3(0.0f);					\
		return;								\
	}

#endif
