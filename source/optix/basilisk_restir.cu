#include "basilisk_common.cuh"

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
		float3 D = fls.point - sh.x;
		float d = length(D);
		D /= d;

		float3 Li = direct_occluded(sh, fls.Le, fls.normal, D, d);

		// Resampling
		float target = length(Li);
		float pdf = fls.pdf;

		float w = (pdf > 0.0f) ? target/pdf : 0.0f;

		reservoir_update(&reservoir, LightSample {
			.value = Li,
			.target = target,
			.type = fls.type,
			.index = fls.index
		}, w, seed);
	}

	// Get final sample and contribution
	LightSample sample = reservoir.sample;
	float W = (sample.target > 0) ? reservoir.weight/(M * sample.target) : 0.0f;

	return W * sample.value;
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

	// TODO: temporal reprojection?

	// Get direct lighting sample
	FullLightSample fls = sample_direct(rp->seed);

	// Compute lighting
	float3 D = fls.point - sh.x;
	float d = length(D);
	D /= d;

	float3 Li = direct_occluded(sh, fls.Le, fls.normal, D, d);

	// Resampling
	float target = length(Li);
	float pdf = fls.pdf;

	float w = (pdf > 0.0f) ? target/pdf : 0.0f;

	reservoir_update(reservoir, LightSample {
		.value = Li,
		.target = target,
		.type = fls.type,
		.index = fls.index
	}, w, rp->seed);

	// Get final sample and contribution
	LightSample sample = reservoir->sample;
	float denominator = reservoir->count * sample.target;
	float W = (sample.target > 0) ? reservoir->weight/denominator : 0.0f;

	return W * sample.value;
}

// Get direct lighting using Spatio-Temporal RIS (ReSTIR)
__device__
float3 direct_lighting_restir(const SurfaceHit &sh, RayPacket *rp)
{
	// Get the reservoir
	LightReservoir *temporal = &parameters.advanced.r_lights[rp->index];
	if (parameters.samples == 0) {
		temporal->sample = LightSample {};
		temporal->count = 0;
		temporal->weight = 0.0f;
		temporal->mis = 0.0f;
	}

	// Get direct lighting sample
	FullLightSample fls = sample_direct(rp->seed);

	// Compute target function (unocculted lighting)
	float3 D = fls.point - sh.x;
	float d = length(D);
	D /= d;

	float3 Li = direct_unoccluded(sh, fls.Le, fls.normal, D, d);

	// Temporal Resampling
	float target = length(Li);
	float pdf = fls.pdf;

	float w = (pdf > 0.0f) ? target/pdf : 0.0f;

	reservoir_update(temporal, LightSample {
		.value = fls.Le,
		.point = fls.point,
		.normal = fls.normal,
		.target = target,
		.type = fls.type,
		.index = fls.index
	}, w, rp->seed);

	// Spatial Resampling
	LightReservoir spatial {
		.sample = LightSample {},
		.count = 0,
		.weight = 0.0f,
		.mis = 0.0f,
	};

	// Add current sample
	int Z = 0;

	{
		// Compute unbiased weight
		LightSample sample = temporal->sample;
		float denominator = temporal->count * sample.target;
		float W = (sample.target > 0) ? temporal->weight/denominator : 0.0f;

		// Compute value and target
		D = sample.point - sh.x;
		d = length(D);
		D /= d;

		float3 Li = direct_occluded(sh, sample.value, sample.normal, D, d);

		// Add to the reservoir
		float target = length(Li);

		float w = target * W * temporal->count;

		spatial.weight += w;

		float p = w/spatial.weight;
		float eta = rand_uniform(rp->seed);

		if (eta < p || spatial.count == 0) {
			spatial.sample = LightSample {
				.value = Li,
				.target = target,
				.type = sample.type,
				.index = sample.index
			};
		}

		spatial.count += temporal->count;
		if (target > 0.0f)
			Z += temporal->count;
	}

	// Sample various neighboring reservoirs
	const int WIDTH = parameters.resolution.x;
	const int HEIGHT = parameters.resolution.y;

	const int SAMPLES = 0;
	const float SAMPLING_RADIUS = min(WIDTH, HEIGHT) * 0.1f;

	int ix = rp->index % WIDTH;
	int iy = rp->index / WIDTH;

	for (int i = 0; i < SAMPLES; i++) {
		// Get offset
		float3 eta = rand_uniform_3f(rp->seed);

		float radius = SAMPLING_RADIUS * sqrt(eta.x);
		float theta = 2.0f * M_PI * eta.y;

		int offx = (int) floorf(radius * cosf(theta));
		int offy = (int) floorf(radius * sinf(theta));

		int nix = ix + offx;
		int niy = iy + offy;

		if (niy < 0 || niy >= HEIGHT || nix < 0 || nix >= WIDTH)
			continue;

		int ni = niy * WIDTH + nix;

		// Get the reservoir
		LightReservoir *reservoir = &parameters.advanced.r_lights[ni];

		// Get sample and resample
		LightSample sample = reservoir->sample;
		float denominator = reservoir->count * sample.target;
		float W = (sample.target > 0) ? reservoir->weight/denominator : 0.0f;

		// Compute value and target
		D = sample.point - sh.x;
		d = length(D);
		D /= d;

		float3 Li = direct_occluded(sh, sample.value, sample.normal, D, d);

		// Add to the reservoir
		float target = length(Li);

		float w = target * W * reservoir->count;

		spatial.weight += w;

		float p = w/spatial.weight;
		if (eta.z < p || spatial.count == 0) {
			spatial.sample = LightSample {
				.value = Li,
				.target = target,
				.type = sample.type,
				.index = sample.index
			};
		}

		spatial.count += reservoir->count;
		if (target > 0.0f)
			Z += reservoir->count;
	}

	// Get final sample's contribution	
	LightSample sample = spatial.sample;
	float denominator = Z * sample.target;
	float W = (sample.target > 0) ? spatial.weight/denominator : 0.0f;

	// Evaluate the integrand
	return W * sample.value;
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
		.mat = material,
		.entering = entering,
		.n = n,
		.wo = wo,
		.x = x,
	};

	// float3 direct = direct_lighting_ris(surface_hit, rp->seed);
	// float3 direct = direct_lighting_temporal_ris(surface_hit, rp);
	float3 direct = direct_lighting_restir(surface_hit, rp);
	if (material.type == Shading::eEmissive)
		direct += material.emission;
	
	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(surface_hit, wi, pdf, out, rp->seed);

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

	rp->position = make_float4(x, 1);
	rp->normal = n;
	rp->albedo = material.diffuse;
	rp->wi = wi;
}
