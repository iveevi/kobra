#include "basilisk_common.cuh"

// Sample from discrete distribution
KCUDA_INLINE static __device__
int sample_discrete(float *pdfs, int num_pdfs, float eta)
{
	float sum = 0.0f;
	for (int i = 0; i < num_pdfs; ++i)
		sum += pdfs[i];
	
	float cdf = 0.0f;
	for (int i = 0; i < num_pdfs; ++i) {
		cdf += pdfs[i] / sum;
		if (eta < cdf)
			return i;
	}

	return num_pdfs - 1;
}

// Reservoir structure
template <class T>
struct Reservoir {
	int M;
	float weight;
	T sample;
};

// Closest hit program for ReSTIR
extern "C" __global__ void __closesthit__restir()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Check if primary ray
	bool primary = (rp->depth == 0);
	
	// TODO: check for light, not just emissive material
	if (hit->material.type == Shading::eEmissive) {
		rp->value = material.emission;
		rp->normal = n;
		rp->albedo = material.diffuse;
		return;
	}
	
	// Offset by normal
	// TODO: use more complex shadow bias functions
	// TODO: an easier check for transmissive objects
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;

	float3 direct = Ld(x, wo, n, material, entering, rp->seed);

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;

	// Resampling Importance Sampling
	constexpr int M = 5;

#if 0

	float3 samples[M];
	float weights[M];
	float wsum = 0;

	for (int i = 0; i < M; i++) {
		// Generate new ray
		Shading out;
		float3 wi;
		float pdf;

		float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
		if (length(f) < 1e-6f)
			continue;

		// Get threshold value for current ray
		trace <eRegular> (x, wi, i0, i1);

		float3 value = f * rp->value * abs(dot(wi, n));

		// RIS computations
		samples[i] = value;
		weights[i] = length(value)/(M * pdf);
		wsum += weights[i];
	}

	// Sample from the distribution
	float eta = fract(random3(rp->seed)).x;
	int index = sample_discrete(&weights[0], M, eta);

	float3 sample = samples[index];
	float W = wsum/length(sample);
	rp->value = direct + W * samples[index];

#elif 0

	// Reservoir sampling
	::Reservoir <float3> reservoir {
		.M = 0,
		.weight = 0.0f,
		.sample = make_float3(0.0f)
	};

	for (int i = 0; i < M; i++) {
		// Generate new ray
		Shading out;
		float3 wi;
		float pdf;

		float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
		if (length(f) < 1e-6f)
			continue;

		// Get threshold value for current ray
		trace <eRegular> (x, wi, i0, i1);

		float3 value = f * rp->value * abs(dot(wi, n));

		// RIS computations
		float w = length(value)/(M * pdf);

		reservoir.weight += w;

		float p = w/reservoir.weight;
		float eta = fract(random3(rp->seed)).x;

		if (eta < p || i == 0)
			reservoir.sample = value;

		reservoir.M++;
	}

	float W = reservoir.weight/length(reservoir.sample);
	rp->value = direct + W * reservoir.sample;

#else

	ReSTIR_Reservoir *reservoir = &parameters.advanced.r_temporal[rp->index];
	if (parameters.samples == 0) {
		reservoir->sample = PathSample {};
		reservoir->weight = 0.0f;
		reservoir->count = 0;
	}

	// Temporal RIS
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);

	trace <eRegular> (x, wi, i0, i1);

	float3 value = rp->value;

	PathSample sample {
		.value = value,
		.position = rp->position,
		.source = x,
		.normal = rp->normal,
		.direction = wi,
		.missed = (rp->miss_depth == 1)
	};

	// TODO: figure out how to do M-capping properly
	// reservoir->count = min(reservoir->count + 1, 20);
	reservoir->count++;

	float target = length(value);

	float w = target/(pdf + 1e-6f);
	reservoir->weight += w;

	float p = w/reservoir->weight;
	float eta = fract(random3(rp->seed)).x;

	if (eta < p || reservoir->count == 1)
		reservoir->sample = sample;

	// float W = reservoir->weight/length(reservoir->sample.value);
	// W /= reservoir->count;

	// Copy temporal to previous temporal
	parameters.advanced.r_temporal_prev[rp->index] = *reservoir;

	// Spatial RIS
	// TODO: persistent reservoirs
	ReSTIR_Reservoir spatial {
		.sample = PathSample {},
		.count = 0,
		.weight = 0.0f,
		.mis = 0.0f
	};

	// TODO: insert current sample at least...?

	const int SPATIAL_SAMPLES = 3;

	int width = parameters.resolution.x;
	int height = parameters.resolution.y;
	
	// TODO: adaptive radius
	const float SAMPLE_RADIUS = 10.0f; // min(width, height)/10.0f;

	int ix = rp->index % width;
	int iy = rp->index / width;

	for (int i = 0; i < SPATIAL_SAMPLES; i++) {
		// Sample pixel in a radius
		float3 eta = fract(random3(rp->seed));

		eta = 2.0f * eta - 1.0f;
		int offx = floorf(eta.x * SAMPLE_RADIUS);
		int offy = floorf(eta.y * SAMPLE_RADIUS);

		int sx = ix + offx;
		int sy = iy + offy;

		/* TODO: is wraparound a good idea?
		if (sx < 0 || sx >= width)
			sx = ix - offx;

		if (sy < 0 || sy >= height)
			sy = iy - offy; */

		if (sx < 0 || sx >= width || sy < 0 || sy >= height)
			continue;

		int index = sx + sy * width;

		// Get the temporal resevoir at that pixel
		ReSTIR_Reservoir *temporal = &parameters.advanced.r_temporal_prev[index];
		if (temporal->count == 0)
			continue;
		
		// Get the sample
		PathSample sample = temporal->sample;

		// Sensible constraints; rays misses or has no contribution
		if (sample.missed || length(sample.value) < 1e-6f)
			continue;

		// Geometry constraints
		float scene_diagonal = length(parameters.voxel.max - parameters.voxel.min);

		float depth_x = length(parameters.camera - x);
		float depth_y = length(parameters.camera - sample.position);

		float angle = acos(dot(n, sample.normal)) * 180.0f/M_PI;
		float depth = abs(depth_x - depth_y)/scene_diagonal;

		// TODO: either this or roughness for GGX
		// NOTE: this depth constraint should depend on the sampling
		// radius; the larger the radius, the more fluctuations
		// there will be in the depth

		// if (angle > 60.0f) // TODO: depth threshold?
		//	continue;

		// Check for occlusion
		float3 L = sample.position - x;
		float d = length(L);
		L /= d;

		// TODO: case when ray missed...
		bool occluded = is_occluded(x + L * 1e-4f, L, d);
		if (occluded)
			continue;

		// Compute weight for the sample for that reservoir
		float W = temporal->weight/length(sample.value);
		W /= temporal->count;

		// TODO: why would this happen?
		// assert(!isnan(W));

		// Compute GRIS weight
		float w = W * length(temporal->sample.value);

		// Compute jacobian of reconnection shift map
		// TODO: what if the ray misses? use only cosine term?
		float3 dir = normalize(sample.position - x);
		float cos_theta_x = abs(dot(dir, sample.normal));
		float cos_theta_y = abs(dot(sample.direction, sample.normal));

		float dist_x = length(sample.position - x);
		float dist_y = length(sample.position - sample.source);

		float dist2_x = dist_x * dist_x;
		float dist2_y = dist_y * dist_y;

		float jacobian = (cos_theta_x/cos_theta_y) * (dist2_y/dist2_x);

		// TODO: jacobian is currently problematic
		w *= jacobian;

		// TODO: when does this happen?
		// if (isnan(w))
		//	assert(isnan(jacobian));

		// assert(!isnan(w));
		// if (isnan(w))
		//	continue;

		// Insert into spatial reservoir
		spatial.weight += w;
		spatial.count++;

		float p = w/spatial.weight;
		if (eta.z < p || spatial.count == 1)
			spatial.sample = temporal->sample;
	}

	// Final sample
	sample = spatial.sample;

	// Compute final GRIS weight
	float W = spatial.weight/length(sample.value);
	W /= spatial.count;

	// assert(!isnan(W));
	if (isnan(W)) {
		// rp->value = make_float3(1, 0, 1);
		// return;
		W = 0.0f;
	}

	// TODO: what to do about entering?
	float3 brdf_value = brdf(material, n, sample.direction, wo, entering, out);
	f = brdf_value * abs(dot(sample.direction, n)) * sample.value;
	rp->value = direct + W * f;

#endif

	// Pass through features
	rp->normal = n;
	rp->albedo = material.diffuse;
}