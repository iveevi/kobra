// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/cuda/brdf.cuh"
#include "../../include/cuda/material.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/optix/parameters.cuh"

using namespace kobra::cuda;
using namespace kobra::optix;

extern "C"
{
	__constant__ kobra::optix::HT_Parameters ht_params;
}

// Local constants
static const float eps = 1e-3f;

// Power heurestic
static const float p = 2.0f;

__device__ float power(float pdf_f, float pdf_g)
{
	float f = pow(pdf_f, p);
	float g = pow(pdf_g, p);

	return f/(f + g);
}

// Check shadow visibility
KCUDA_INLINE __device__
bool shadow_visibility(float3 origin, float3 dir, float R)
{
	bool vis = false;

	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	optixTrace(ht_params.traversable,
		origin, dir,
		0, R - 0.01f, 0,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
			| OPTIX_RAY_FLAG_DISABLE_ANYHIT
			| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		ht_params.instances, 0, 1,
		j0, j1
	);

	return vis;
}

// Direct lighting for specific types of lights
template <class Light>
__device__ float3 Ld_light(const Light &light, float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	float3 contr_nee {0.0f};
	float3 contr_brdf {0.0f};

	// NEE
	float3 lpos = sample_area_light(light, seed);
	float3 wi = normalize(lpos - x);
	float R = length(lpos - x);

	float3 f = brdf(mat, n, wi, wo, entering, mat.type) * abs(dot(n, wi));

	float ldot = abs(dot(light.normal(), wi));
	if (ldot > 1e-6) {
		float pdf_light = (R * R)/(light.area() * ldot);

		// TODO: how to decide ray type for this?
		float pdf_brdf = pdf(mat, n, wi, wo, entering, mat.type);

		bool vis = shadow_visibility(x + n * eps, wi, R);
		if (pdf_light > 1e-9 && vis) {
			float weight = power(pdf_light, pdf_brdf);
			float3 intensity = light.intensity;
			contr_nee += weight * f * intensity/pdf_light;
		}
	}

	// BRDF
	Shading out;
	float pdf_brdf;

	f = eval(mat, n, wo, entering, wi, pdf_brdf, out, seed) * abs(dot(n, wi));
	if (length(f) < 1e-6f)
		return contr_nee;

	float pdf_light = 0.0f;

	// TODO: need to check intersection for lights specifically (and
	// arbitrary ones too?)
	float ltime = light.intersects(x, wi);
	if (ltime <= 0.0f)
		return contr_nee;
	
	float weight = 1.0f;
	if (out & eTransmission) {
		return contr_nee;
		// pdf_light = (R * R)/(light.area() * ldot);
	} else {
		R = ltime;
		pdf_light = (R * R)/(light.area() * abs(dot(light.normal(), wi)));
		weight = power(pdf_brdf, pdf_light);
	};

	// TODO: shoot shadow ray up to R
	if (pdf_light > 1e-9 && pdf_brdf > 1e-9) {
		float3 intensity = light.intensity;
		contr_brdf += weight * f * intensity/pdf_brdf;
	}

	return contr_nee + contr_brdf;
}

// Trace ray into scene and get relevant information
__device__ float3 Ld(float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	int quad_count = ht_params.lights.quad_count;
	int tri_count = ht_params.lights.triangle_count;

	if (quad_count == 0 && tri_count == 0)
		return make_float3(0.0f);

	// TODO: multiply result by # of total lights

	// Random area light for NEE

// #define LIGHT_SAMPLES 5

#ifdef LIGHT_SAMPLES

	float3 contr {0.0f};

	for (int k = 0; k < LIGHT_SAMPLES; k++) {
		random3(seed);
		unsigned int i = seed.x * (hit_data->n_quad_lights + hit_data->n_tri_lights);
		i = min(i, hit_data->n_quad_lights + hit_data->n_tri_lights - 1);

		if (i < hit_data->n_quad_lights) {
			QuadLight light = hit_data->quad_lights[i];
			contr += Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
		} else {
			TriangleLight light = hit_data->tri_lights[i - hit_data->n_quad_lights];
			contr += Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
		}
	}

	return contr/LIGHT_SAMPLES;

#else 

	random3(seed);
	unsigned int i = seed.x * (quad_count + tri_count);
	i = min(i, quad_count + tri_count - 1);

	if (i < quad_count) {
		QuadLight light = ht_params.lights.quads[i];
		return Ld_light(light, x, wo, n, mat, entering, seed);
	}

	TriangleLight light = ht_params.lights.triangles[i - quad_count];
	return Ld_light(light, x, wo, n, mat, entering, seed);

#endif

}

// Ray packet data
struct RayPacket {
	float3	throughput;
	
	float3	value;
	float3	seed;
	float	ior;

	int	depth;
};

// Ray generation kernel
extern "C" __global__ void __raygen__rg()
{
	// TODO: perform the first direct lihgting in a CUDA kernel,
	// then pass the position, value and direction to the raygen kernel
	// to reduce the stack size

	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

	int object_index = tex2D <int> (
		ht_params.ids, idx.x,
		ht_params.resolution.y - idx.y
	);

	// ht_params.color_buffer[index] = float4 {object_index/255.0f, 0.0f, 0.0f, 1.0f};
	// return;

	if (object_index <= 0) {
		const float3 U = ht_params.cam_u;
		const float3 V = ht_params.cam_v;
		const float3 W = ht_params.cam_w;

		// Compute ray origin and direction
		float2 d = 2.0f * make_float2(
			float(idx.x + 0.5f)/float(ht_params.resolution.x),
			float(idx.y + 0.5f)/float(ht_params.resolution.y)
		) - 1.0f;

		float3 dir = normalize(d.x * U + d.y * V + W);
	
		float u = atan2(dir.x, dir.z) / (2.0f * M_PI) + 0.5f;
		float v = asin(dir.y) / M_PI + 0.5f;

		float4 c = tex2D <float4> (ht_params.envmap, u, v);

		ht_params.color_buffer[index] = c;
		return;
	}

	// Calculate UV coordinates
	float2 uv = make_float2(
		(float) idx.x/(float) ht_params.resolution.x,
		(float) idx.y/(float) ht_params.resolution.y
	);

	uv.y = 1.0f - uv.y;

	// Extract the initial bounce information
	float3 x = make_float3(tex2D <float4> (ht_params.positions, uv.x, uv.y));
	float3 n = make_float3(tex2D <float4> (ht_params.normals, uv.x, uv.y));
	float3 wo = normalize(ht_params.camera - x);

	n = normalize(n);
	if (dot(n, wo) < 0.0f)
		n = -n;

	// TODO: why would the following condition be triggered
	if(isnan(n.x) || isnan(n.y) || isnan(n.z)) {
		ht_params.color_buffer[index] = float4 {0, 0, 0, 1};
		return;
	}

	Material mat {};
	mat.diffuse = make_float3(tex2D <float4> (ht_params.albedo, uv.x, uv.y));
	mat.specular = make_float3(tex2D <float4> (ht_params.specular, uv.x, uv.y));

	float4 extra = tex2D <float4> (ht_params.extra, uv.x, uv.y);

	mat.shininess = extra.x;
	mat.roughness = extra.y;
	mat.type = eDiffuse;

	// Store color
	float3 seed {float(idx.x), float(idx.y), ht_params.time};
	float3 direct = Ld(x, wo, n, mat, true, seed);
	
	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(mat, n, wo, true, wi, pdf, out,seed);

	// Store the result
	RayPacket rp {
		.throughput = f * abs(dot(wi, n))/pdf,
		.value = direct,
		.seed = seed,
		.ior = 1, // TODO: get from textures
		.depth = 1,
	};

	// Pack the ray packet
	unsigned int i0, i1;
	pack_pointer(&rp, i0, i1);
	
	// Trace to get multibounce global illumination
	float3 offset = 1e-3f * n;
	if (out & Shading::eTransmission)
		offset = 1e-3f * wi;

	if (length(f) > 1e-6) {
		optixTrace(ht_params.traversable,
			x + offset, wi,
			0.0f, 1e16f, 0.0f,
			OptixVisibilityMask(0b1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0, 0, 0,
			i0, i1
		);
	}

	// Finally, store the result
	ht_params.color_buffer[index] = make_float4(rp.value);
}

struct mat3 {
	// Column major
	float m[9];

	__device__ __forceinline__ mat3() {}

	__device__ __forceinline__ mat3(float3 c1, float3 c2, float3 c3) {
		// Store in column major order
		m[0] = c1.x; m[3] = c2.x; m[6] = c3.x;
		m[1] = c1.y; m[4] = c2.y; m[7] = c3.y;
		m[2] = c1.z; m[5] = c2.z; m[8] = c3.z;
	}
};

__device__ __forceinline__ float3 operator*(mat3 m, float3 v)
{
	return make_float3(
		m.m[0] * v.x + m.m[3] * v.y + m.m[6] * v.z,
		m.m[1] * v.x + m.m[4] * v.y + m.m[7] * v.z,
		m.m[2] * v.x + m.m[5] * v.y + m.m[8] * v.z
	);
}

// Interpolate triangle values
template <class T>
__device__ T interpolate(T *arr, uint3 triagle, float2 bary)
{
	T a = arr[triagle.x];
	T b = arr[triagle.y];
	T c = arr[triagle.z];

	return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

// Sample from a texture
static __forceinline__ __device__ float4 sample_texture
		(Hit *hit_data, cudaTextureObject_t tex, uint3 triangle, float2 bary)
{
	float2 uv = interpolate(hit_data->texcoords, triangle, bary);
	return tex2D <float4> (tex, uv.x, 1 - uv.y);
}

// Calculate hit normal
static __forceinline__ __device__ float3 calculate_normal
		(Hit *hit_data, uint3 triangle, float2 bary,
		 bool &entering)
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
		float4 n4 = sample_texture(hit_data,
			hit_data->textures.normal,
			triangle, bary
		);

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
__device__ void calculate_material
		(Hit *hit_data,
		Material &mat,
		uint3 triangle, float2 bary)
{
	if (hit_data->textures.has_diffuse) {
		mat.diffuse = make_float3(
			sample_texture(hit_data,
				hit_data->textures.diffuse,
				triangle, bary
			)
		);
	}

	if (hit_data->textures.has_roughness) {
		mat.roughness = sample_texture(hit_data,
			hit_data->textures.roughness,
			triangle, bary
		).x;
	}
}

#define MAX_DEPTH 3

// Closest hit kernel
extern "C" __global__ void __closesthit__ch()
{
	// Get payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_point <RayPacket> (i0, i1);

	if (rp->depth > MAX_DEPTH)
		return;
	
	// Get data from the SBT
	Hit *hit = reinterpret_cast <Hit *> (optixGetSbtDataPointer());

	// Calculate relevant data for the hit
	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	uint3 triangle = hit->triangles[primitive_index];

	Material material = hit->material;

	// TODO: check for light, not just emissive material
	if (hit->material.type == Shading::eEmissive) {
		rp->value += rp->throughput * material.emission;
		rp->throughput = {0, 0, 0};
		return;
	}
	
	calculate_material(hit, material, triangle, bary);

	bool entering;
	float3 wo = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit, triangle, bary, entering);
	float3 x = interpolate(hit->vertices, triangle, bary);

	float3 direct = Ld(x, wo, n, material, entering, rp->seed);
	rp->value += rp->throughput * direct;

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
	if (length(f) < 1e-6f)
		return;
	
	float3 T = f * abs(dot(wi, n))/pdf;

	// Russian roulette
	float p = max(rp->throughput.x, max(rp->throughput.y, rp->throughput.z));
	float q = 1 - min(1.0f, p);

	if (fract(rp->seed.x) < q)
		return;

	rp->throughput *= T/(1 - q);

	// Recursive raytrace
	float3 offset = 1e-3f * n;
	if (out & Shading::eTransmission)
		offset = 1e-3f * wi;

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;

	// Recurse
	optixTrace(ht_params.traversable,
		x + offset, wi,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 0, 0,
		i0, i1
	);
}

extern "C" __global__ void __closesthit__shadow() {}

// Miss kernel
extern "C" __global__ void __miss__ms()
{
	// Get direction
	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y)/M_PI + 0.5f;

	float4 c = tex2D <float4> (ht_params.envmap, u, v);

	// Transfer to payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_point <RayPacket> (i0, i1);

	rp->value += rp->throughput * make_float3(c);
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_point <bool> (i0, i1);
	*vis = true;
}
