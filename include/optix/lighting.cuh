#ifndef KOBRA_OPTIX_LIGHTING_H_
#define KOBRA_OPTIX_LIGHTING_H_

// Engine headers
#include "../cuda/material.cuh"
#include "../cuda/brdf.cuh"

// Forward declarations
bool is_occluded(float3, float3, float);

namespace kobra {

namespace optix {

// Power heurestic
static const float p = 2.0f;

KCUDA_INLINE KCUDA_HOST_DEVICE
float power(float pdf_f, float pdf_g)
{
	float f = pow(pdf_f, p);
	float g = pow(pdf_g, p);

	return f/(f + g);
}

// Direct lighting for specific types of lights
template <class Light>
KCUDA_HOST_DEVICE
float3 Ld_light(const Light &light, float3 x, float3 wo, float3 n,
		cuda::Material mat, bool entering, float3 &seed)
{
	static const float eps = 0.05f;

	float3 contr_nee {0.0f};
	float3 contr_brdf {0.0f};

	// NEE
	float3 lpos = sample_area_light(light, seed);
	float3 wi = normalize(lpos - x);
	float R = length(lpos - x);

	float3 f = cuda::brdf(mat, n, wi, wo, entering, mat.type) * abs(dot(n, wi));

	float ldot = abs(dot(light.normal(), wi));
	if (ldot > 1e-6) {
		float pdf_light = (R * R)/(light.area() * ldot);

		// TODO: how to decide ray type for this?
		float pdf_brdf = cuda::pdf(mat, n, wi, wo, entering, mat.type);

		bool vis = is_occluded(x + n * eps, wi, R);
		if (pdf_light > 1e-9 && !vis) {
			float weight = power(pdf_light, pdf_brdf);
			float3 intensity = light.intensity;
			contr_nee += weight * f * intensity/pdf_light;
		}
	}

	// BRDF
	Shading out;
	float pdf_brdf;

	f = cuda::eval(mat, n, wo, entering, wi, pdf_brdf, out, seed) * abs(dot(n, wi));
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

}

}

#endif
