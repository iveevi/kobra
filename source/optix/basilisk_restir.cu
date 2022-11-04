#include "basilisk_common.cuh"

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

__device__ __forceinline__
FullLightSample sample_direct(Seed seed)
{
	// TODO: plus envmap
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.triangle_count;

	unsigned int total_lights = quad_count + tri_count;
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
	}

	return sample;
}

// Compute direct lighting for a given sample
__device__ __forceinline__
float3 direct_at(const SurfaceHit &sh, const FullLightSample &fls, float3 D, float d)
{
	// Assume that the light is visible
	float3 rho = cuda::brdf(sh.mat,
		sh.n, D, sh.wo,
		sh.entering, sh.mat.type
	);

	float ldot = abs(dot(fls.normal, D));
	float geometric = ldot * abs(dot(sh.n, D))/(d * d);

	return rho * fls.Le * geometric;
}

// Get direct lighting using RIS
__device__
float3 direct_lighting_ris(const SurfaceHit &sh, Seed seed)
{
	const int M = 10;

	LightReservoir reservoir {
		.sample = LightSample {},
		.count = 0,
		.weight = 0.0f,
		.mis = 0.0f,
	};

	for (int k = 0; k < M; k++) {
		// Get direct lighting sample
		FullLightSample fls = sample_direct(seed);

		// Compute lighting
		// TODO: method
		float3 D = fls.point - sh.x;
		float d = length(D);
		D /= d;

		bool occluded = is_occluded(sh.x, D, d);
		
		float3 Li = make_float3(0.0f);
		if (!occluded)
			Li = direct_at(sh, fls, D, d);

		// Resampling
		float target = length(Li);
		float pdf = fls.pdf;

		float w = (pdf > 0.0f) ? target/pdf : 0.0f;

		reservoir.weight += w;

		float p = w/reservoir.weight;
		float eta = rand_uniform(seed);

		if (eta < p || reservoir.count == 0) {
			reservoir.sample = LightSample {
				.contribution = Li,
				.target = target,
				.type = fls.type,
				.index = fls.index
			};
		}

		reservoir.count++;
	}

	// Get final sample and contribution
	LightSample sample = reservoir.sample;
	float W = (sample.target > 0) ? reservoir.weight/(M * sample.target) : 0.0f;

	return W * sample.contribution;
}

// Get direct lighting using Temporal RIS
__device__
float3 direct_lighting_temporal_ris(const SurfaceHit &sh, RayPacket *rp)
{
	// Get the reservoir
	LightReservoir *reservoir = &parameters.advanced.r_lights[rp->index];
	if (parameters.samples == 0) {
		reservoir->sample = LightSample {};
		reservoir->count = 0;
		reservoir->weight = 0.0f;
		reservoir->mis = 0.0f;
	}

	// TODO: reset reservoir if needed
	// TODO: temporal reprojection?

	// Get direct lighting sample
	FullLightSample fls = sample_direct(rp->seed);

	// Compute lighting
	float3 D = fls.point - sh.x;
	float d = length(D);
	D /= d;

	bool occluded = is_occluded(sh.x, D, d);

	float3 Li = make_float3(0.0f);
	if (!occluded)
		Li = direct_at(sh, fls, D, d);

	// Resampling
	float target = length(Li);
	float pdf = fls.pdf;

	float w = (pdf > 0.0f) ? target/pdf : 0.0f;

	reservoir->weight += w;

	float p = w/reservoir->weight;
	float eta = rand_uniform(rp->seed);

	if (eta < p || reservoir->count == 0) {
		reservoir->sample = LightSample {
			.contribution = Li,
			.target = target,
			.type = fls.type,
			.index = fls.index
		};
	}

	reservoir->count++;

	// Get final sample and contribution
	LightSample sample = reservoir->sample;
	float denominator = reservoir->count * sample.target;
	float W = (sample.target > 0) ? reservoir->weight/denominator : 0.0f;

	return W * sample.contribution;
}

// Closest hit program for ReSTIR
extern "C" __global__ void __closesthit__restir()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Offset by normal
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;

	// Construct SurfaceHit instance for lighting calculations
	SurfaceHit surface_hit {
		.x = x,
		.wo = wo,
		.n = n,
		.mat = material,
		.entering = entering
	};

	// float3 direct = direct_lighting_ris(surface_hit, rp->seed);
	float3 direct = direct_lighting_temporal_ris(surface_hit, rp);
	if (material.type == Shading::eEmissive)
		direct += material.emission;
	
	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);

	// Get threshold value for current ray
	float3 T = f * abs(dot(wi, n))/pdf;

	// Update for next ray
	rp->ior = material.refraction;
	rp->pdf *= pdf;
	rp->depth++;
	
	// Trace the next ray
	float3 indirect = make_float3(0.0f);
	if (pdf > 0) {
		trace <eRegular> (x, wi, i0, i1);
		indirect = rp->value;
	}

	// Update the value
	rp->value = direct;
	if (pdf > 0)
		rp->value += T * indirect;

	rp->position = x;
	rp->normal = n;
	rp->albedo = material.diffuse;
	rp->wi = wi;
}
