#include "wadjet_common.cuh"

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
	// if (length(f) < 1e-6f)
	//	return;

	trace <eRegular> (x, wi, i0, i1);

	float3 value = f * rp->value * abs(dot(wi, n));

	PathSample sample {
		.value = value,
	};

	// reservoir->count = min(reservoir->count + 1, 20);
	reservoir->count++;
	float w = length(value)/pdf;

	reservoir->weight += w;

	float p = w/reservoir->weight;
	float eta = fract(random3(rp->seed)).x;

	if (eta < p || reservoir->count == 1)
		reservoir->sample = sample;

	float W = reservoir->weight/length(reservoir->sample.value);
	W /= reservoir->count;

	rp->value = direct + W * reservoir->sample.value;

#endif

	// Pass through features
	rp->normal = n;
	rp->albedo = material.diffuse;
}
