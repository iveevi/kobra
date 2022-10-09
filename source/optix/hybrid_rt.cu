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
__device__ bool shadow_visibility(float3 origin, float3 dir, float R)
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

// Ray generation kernel
extern "C" __global__ void __raygen__rg()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

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

	if (dot(n, wo) < 0.0f)
		n = -n;

	Material mat {};
	mat.diffuse = make_float3(tex2D <float4> (ht_params.albedo, uv.x, uv.y));
	mat.specular = make_float3(tex2D <float4> (ht_params.specular, uv.x, uv.y));

	float4 extra = tex2D <float4> (ht_params.extra, uv.x, uv.y);

	mat.shininess = extra.x;
	mat.roughness = extra.y;
	mat.type = eDiffuse;

	// Store color
	float3 seed {float(idx.x), float(idx.y), ht_params.time};

	ht_params.color_buffer[index] = make_float4(Ld(x, wo, n, mat, true, seed));
}

// Closest hit kernel
extern "C" __global__ void __closesthit__ch()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

	// Store color
	ht_params.color_buffer[index] = {1, 0, 0, 1};
}

extern "C" __global__ void __closesthit__shadow() {}

// Miss kernel
extern "C" __global__ void __miss__ms()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

	// Store color
	ht_params.color_buffer[index] = {0, 0, 1, 1};
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_point <bool> (i0, i1);
	*vis = true;
}
