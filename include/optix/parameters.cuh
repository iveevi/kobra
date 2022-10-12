#ifndef KOBRA_OPTIX_PARAMETERS_H_
#define KOBRA_OPTIX_PARAMETERS_H_

// Engine headers
#include "../cuda/material.cuh"
#include "../cuda/math.cuh"
#include "../cuda/random.cuh"
#include "reservoir.cuh"

namespace kobra {

namespace optix {

KCUDA_INLINE KCUDA_HOST_DEVICE
float intersects_triangle
		(float3 v1, float3 v2, float3 v3,
		float3 origin, float3 dir)
{
	float3 e1 = v2 - v1;
	float3 e2 = v3 - v1;
	float3 s1 = cross(dir, e2);
	float divisor = dot(s1, e1);
	if (divisor == 0.0)
		return -1;
	float3 s = origin - v1;
	float inv_divisor = 1.0 / divisor;
	float b1 = dot(s, s1) * inv_divisor;
	if (b1 < 0.0 || b1 > 1.0)
		return -1;
	float3 s2 = cross(s, e1);
	float b2 = dot(dir, s2) * inv_divisor;
	if (b2 < 0.0 || b1 + b2 > 1.0)
		return -1;
	float t = dot(e2, s2) * inv_divisor;
	return t;
}

// Light type
struct QuadLight {
	float3 a;
	float3 ab;
	float3 ac;
	float3 intensity;

	KCUDA_INLINE KCUDA_HOST_DEVICE
	float area() const {
		return length(cross(ab, ac));
	}

	KCUDA_INLINE KCUDA_HOST_DEVICE
	float3 normal() const {
		return normalize(cross(ab, ac));
	}

	KCUDA_INLINE KCUDA_HOST_DEVICE
	float intersects(float3 origin, float3 dir) const {
		float3 v1 = a;
		float3 v2 = a + ab;
		float3 v3 = a + ac;
		float3 v4 = a + ab + ac;

		float t1 = intersects_triangle(v1, v2, v3, origin, dir);
		float t2 = intersects_triangle(v2, v3, v4, origin, dir);

		if (t1 < 0.0 && t2 < 0.0)
			return -1.0;
		if (t1 < 0.0)
			return t2;
		if (t2 < 0.0)
			return t1;

		return (t1 < t2) ? t1 : t2;
	}
};

// Triangular area light
struct TriangleLight {
	float3 a;
	float3 ab;
	float3 ac;
	float3 intensity;

	KCUDA_INLINE KCUDA_HOST_DEVICE
	float area() const {
		return length(cross(ab, ac)) * 0.5;
	}

	KCUDA_INLINE KCUDA_HOST_DEVICE
	float3 normal() const {
		return normalize(cross(ab, ac));
	}

	KCUDA_INLINE KCUDA_HOST_DEVICE
	float intersects(float3 origin, float3 dir) const {
		float3 v1 = a;
		float3 v2 = a + ab;
		float3 v3 = a + ac;

		return intersects_triangle(v1, v2, v3, origin, dir);
	}
};

// Sampling methods
// TODO: move else where
KCUDA_INLINE KCUDA_HOST_DEVICE
float3 sample_area_light(QuadLight light, float3 &seed)
{
	float3 rand = random3(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);
	return light.a + u * light.ab + v * light.ac;
}

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 sample_area_light(TriangleLight light, float3 &seed)
{
	float3 rand = random3(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);
	
	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}
	
	return light.a + u * light.ab + v * light.ac;
}

// Reservoir sample for ReSTIR
struct PathSample {
	float3 value;
	float3 dir;

	float3 p_pos;
	float3 p_normal;

	float3 s_pos;
	float3 s_normal;

	bool missed;
};

using ReSTIR_Reservoir = Reservoir <PathSample>;

// Hit data record
struct Hit {
	// Mesh data
	float2			*texcoords;
	float3			*vertices;
	uint3			*triangles;

	float3			*normals;
	float3			*tangents;
	float3			*bitangents;

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

// Kernel-common parameters for hybrid tracer
struct HT_Parameters {
	// Image resolution
	uint2 resolution;

	// Camera position
	float3 camera;

	float3 cam_u;
	float3 cam_v;
	float3 cam_w;

	// Time
	float time;

	// Accumulation status
	bool accumulate;
	int samples;

	// Scene information
	OptixTraversableHandle traversable;

	int instances;

	// G-buffer textures
	cudaTextureObject_t positions;
	cudaTextureObject_t normals;
	cudaTextureObject_t ids;

	cudaTextureObject_t albedo;
	cudaTextureObject_t specular;
	cudaTextureObject_t extra;

	cudaTextureObject_t envmap;

	// Lights
	struct {
		QuadLight *quads;
		TriangleLight *triangles;

		uint quad_count;
		uint triangle_count;
	} lights;

	// Reservoirs and advanced sampling strategies
	struct {
		ReSTIR_Reservoir *r_temporal;
		ReSTIR_Reservoir *r_temporal_prev;
		
		ReSTIR_Reservoir *r_spatial;
		ReSTIR_Reservoir *r_spatial_prev;
	} advanced;

	// Output buffers
	float4 *color_buffer;
};

// Kernel-common parameters for Wadjet path tracer
struct WadjetParameters {
	// Image resolution
	uint2 resolution;

	// Camera position
	float3 camera;

	float3 cam_u;
	float3 cam_v;
	float3 cam_w;

	// Time
	float time;

	// Accumulation status
	bool accumulate;
	int samples;

	// Scene information
	OptixTraversableHandle traversable;

	int instances;

	// Textures
	cudaTextureObject_t envmap;

	// Lights
	struct {
		QuadLight *quads;
		TriangleLight *triangles;

		uint quad_count;
		uint triangle_count;
	} lights;

	// Reservoirs and advanced sampling strategies
	struct {
		ReSTIR_Reservoir *r_temporal;
		ReSTIR_Reservoir *r_temporal_prev;
		
		ReSTIR_Reservoir *r_spatial;
		ReSTIR_Reservoir *r_spatial_prev;

		float *sampling_radii;
	} advanced;

	// Output buffers
	float4 *color_buffer;
};

}

}

#endif
