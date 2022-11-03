#include "basilisk_common.cuh"

// Get direct lighting using ReSTIR
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
		DirectLightSample dls = sample_direct(sh, seed);

		float target = length(dls.Li);
		float pdf = dls.pdf;

		float w = (pdf > 0.0f) ? target/pdf : 0.0f;

		reservoir.weight += w;

		float p = w/reservoir.weight;
		float eta = rand_uniform(seed);

		if (eta < p || reservoir.count == 0) {
			reservoir.sample = LightSample {
				.contribution = dls.Li,
				.target = target,
				.type = dls.type,
				.index = dls.index
			};
		}

		reservoir.count++;
	}

	// Get final sample and contribution
	LightSample sample = reservoir.sample;
	float W = (sample.target > 0) ? reservoir.weight/(M * sample.target) : 0.0f;

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

	float3 direct = direct_lighting_ris(surface_hit, rp->seed);
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
