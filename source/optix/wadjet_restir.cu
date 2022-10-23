#include "wadjet_common.cuh"

// Weighting function
KCUDA_INLINE static __device__
float weight_kernel(const PathSample &sample)
{
	return length(sample.value);
}

/* Temporal resampling
KCUDA_INLINE __device__
float3 temporal_reuse(RayPacket *rp, const PathSample &sample, float weight)
{
	// Get reservoir
	auto &r_temporal = parameters.advanced.r_temporal[rp->index];

	// Proceed to add the current sample to the reservoir
	r_temporal.update(sample, weight);
	r_temporal.W = r_temporal.weight/(
		r_temporal.count * weight_kernel(r_temporal.sample)
		+ 1e-6f
	);

	// Get resampled value
	return r_temporal.sample.value;
}

// Spatiotemporal resampling
KCUDA_INLINE __device__
float3 spatiotemporal_reuse(RayPacket *rp, float3 x, float3 n)
{
	// X and Y of the pixel
	int ix = rp->index % parameters.resolution.x;
	int iy = rp->index / parameters.resolution.x;

	// Then use spatial resampling
	auto &r_spatial = parameters.advanced.r_spatial[rp->index];
	auto &r_temporal = parameters.advanced.r_temporal[rp->index];
	auto &s_radius = parameters.advanced.sampling_radii[rp->index];
	// s_radius = 10.0f;

	int Z = 0;
	int success = 0;

	r_spatial.merge(r_temporal, weight_kernel(r_temporal.sample));
	Z += r_temporal.count;
	
	const int SPATIAL_SAMPLES = (r_spatial.count < 250) ? 9 : 3;
	for (int i = 0; i < SPATIAL_SAMPLES; i++) {
		// Generate random neighboring pixel
		random3(rp->seed);

		float radius = s_radius * fract(rp->seed.x);
		float angle = 2 * M_PI * fract(rp->seed.y);

		int ny = iy + radius * sin(angle);
		int nx = ix + radius * cos(angle);
		
		if ((nx < 0 || nx >= parameters.resolution.x)
				|| (ny < 0 || ny >= parameters.resolution.y))
			continue;

		int nindex = ny * parameters.resolution.x + nx;

		// Get the appropriate reservoir
		auto *reservoir = &parameters.advanced.r_spatial_prev[nindex];
		if (reservoir->count > 50)
			reservoir = &parameters.advanced.r_temporal_prev[nindex];

		if (reservoir->count == 0)
			continue;

		// Get information relative to sample
		auto &sample = reservoir->sample;

		// Check geometry similarity
		float depth_x = length(x - parameters.camera);
		float depth_s = length(sample.p_pos - parameters.camera);

		float theta = 180 * acos(dot(n, sample.s_normal))/M_PI;
		float ndepth = abs(depth_x - depth_s)/max(depth_x, depth_s);

		if (angle > 25 || ndepth > 0.1)
			continue;

		// Check if the sample is visible
		float3 direction = normalize(sample.p_pos - x);
		float distance = length(sample.p_pos - x);

		bool occluded;
		if (sample.missed)
			occluded = is_occluded(x + sample.dir * eps, sample.dir, 1e6);
		else
			occluded = is_occluded(x + direction * eps, direction, distance);

		if (occluded)
			continue;

		// Compute Jacobian
		float3 xq_1 = sample.p_pos;
		float3 xq_2 = sample.s_pos;
		float3 xr_1 = x;

		float3 v_r = xr_1 - xq_2;
		float3 v_q = xq_1 - xq_2;

		float d_r = length(v_r);
		float d_q = length(v_q);

		v_r /= d_r;
		v_q /= d_q;

		float phi_r = acos(dot(sample.s_normal, v_r));
		float phi_q = acos(dot(sample.s_normal, v_q));

		float J = abs(phi_r/phi_q) * (d_q * d_q)/(d_r * d_r);

		// If conditions are sufficient, merge reservoir
		r_spatial.merge(
			*reservoir,
			weight_kernel(reservoir->sample)/J
		);

		Z += reservoir->count;
		success++;
	}

	// Compute final weight
	r_spatial.W = r_spatial.weight/(
		Z * weight_kernel(r_spatial.sample)
		+ 1e-6f
	);

	// Reduce radius if no samples were found
	if (success == 0)
		s_radius = max(s_radius * 0.5f, 3.0f);

	// Get resampled value
	return r_spatial.sample.value;
} */

// Closest hit program for ReSTIR
extern "C" __global__ void __closesthit__restir()
{
	/* Get payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_pointer <RayPacket> (i0, i1);
	
	if (rp->depth > MAX_DEPTH)
		return; */

	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Check if primary ray
	bool primary = (rp->depth == 0);
	
	// TODO: check for light, not just emissive material
	if (hit->material.type == Shading::eEmissive) {
		rp->value = material.emission;
		return;
	}
	
	// Offset by normal
	// TODO: use more complex shadow bias functions
	// TODO: an easier check for transmissive objects
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;

	float3 direct = Ld(x, wo, n, material, entering, rp->seed);

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
	if (length(f) < 1e-6f)
		return;

	// Get threshold value for current ray
	float3 T = f * abs(dot(wi, n))/pdf;

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;

	// Recurse
	trace <eRegular> (x, wi, i0, i1);

	// Post: advanced sampling techniques if any
	float3 indirect = rp->value;

	// ReSTIR GI
	float max_radius = min(
		parameters.resolution.x,
		parameters.resolution.y
	)/10.0f;

	// TODO: temporal reporjection at some point
	if (parameters.samples == 0) {
		auto *r_temporal = &parameters.advanced.r_temporal[rp->index];
		// auto *r_spatial = &parameters.advanced.r_spatial[rp->index];
		auto *s_radius = &parameters.advanced.sampling_radii[rp->index];

		// Reset for motion
		// r_temporal->reset();
		r_temporal->count = 0;
		r_temporal->weight = 0.0f;
		r_temporal->mis = 0.0f;

		// r_spatial->reset();
		*s_radius = max_radius;
	}

	ReSTIR_Reservoir *r_temporal = &parameters.advanced.r_temporal[rp->index];

#if 0

	// Generate sample and weight
	PathSample sample {
		.value = f * rp->value * abs(dot(wi, n)),
		.dir = wi,
		.p_pos = x,
		.p_normal = n,
		.s_pos = rp->position,
		.s_normal = rp->normal,
		.target = length(f * rp->value * abs(dot(wi, n))),
		.missed = rp->missed,
	};

	// r_temporal->update(sample, weight);

	// NOTE: why is it incorrect to multiply pdf by path pdf?
	// pdf = rp->pdf;
	r_temporal->mis += pdf + 1e-4f;

	float mis = pdf/r_temporal->mis;
	float weight = mis * (sample.target/pdf);

	r_temporal->weight += weight;
	r_temporal->count = min(r_temporal->count + 1, 20);
	// r_temporal->count++;

	float q = weight/r_temporal->weight;
	float e = fract(random3(rp->seed)).x;
	if (e < q)
		r_temporal->sample = sample;

	sample = r_temporal->sample;

	int count = r_temporal->count;
	// float W = (r_temporal->weight/count)/(sample.target + 1e-6f);
	float W = r_temporal->weight/(sample.target + 1e-4f);

	float3 brdf_value = brdf(material, n, sample.dir, wo, entering, material.type);
	rp->value = direct + sample.value * W;
	
	/* reset
	r_temporal->count = 0;
	r_temporal->weight = 0.0f;
	r_temporal->mis = 0.0f; */

#else

	float3 value = direct + T * indirect;

	PathSample sample {
		.value = value,
		.dir = wi,
		.target = length(value)
	};

	// Update reservoir
	r_temporal->mis += pdf + 1e-4f;

	float mis = pdf/r_temporal->mis;
	float weight = mis * (sample.target/pdf);

	r_temporal->weight += weight;
	r_temporal->count = min(r_temporal->count + 1, 20);
	// r_temporal->count++;

	float q = weight/r_temporal->weight;
	float e = fract(random3(rp->seed)).x;

	if (e < q)
		r_temporal->sample = sample;

	sample = r_temporal->sample;

	float W = r_temporal->weight/(sample.target + 1e-4f);

	rp->value = sample.value * W;

#endif

}

// rp->value = make_float3(1/(1 + W));

// TODO: spatial sampling if samples > 0

/* First actually update the temporal reservoir
temporal_reuse(rp, sample, weight);

auto &r_temporal = parameters.advanced.r_temporal[rp->index];
sample = r_temporal.sample;

float3 brdf = kobra::cuda::brdf(material, n,
	sample.dir, wo,
	entering, material.type
);

rp->value = direct + brdf * sample.value * r_temporal.W *
abs(dot(sample.dir, n)); */

/* if (parameters.samples > 0) {
	// Then use spatiotemporal resampling
	indirect = spatiotemporal_reuse(rp, x, n);

	auto &r_spatial = parameters.advanced.r_temporal[rp->index];

	// TODO: recalculate value of f, using brdf...
	float3 brdf = kobra::cuda::brdf(material, n,
		r_spatial.sample.dir, wo,
		entering, material.type
	);

	float pdf = kobra::cuda::pdf(material, n,
		r_spatial.sample.dir, wo,
		entering, material.type
	);

	rp->value = direct + brdf * r_spatial.sample.value *
		r_spatial.W * abs(dot(r_spatial.sample.dir, n));
} */
