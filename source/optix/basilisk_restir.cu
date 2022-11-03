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

// Closest hit program for ReSTIR
extern "C" __global__ void __closesthit__restir()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Offset by normal
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;

	float3 direct = Ld <true> (x, wo, n, material, entering, rp->seed);
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
