#ifndef KOBRA_CUDA_BRDF_H_
#define KOBRA_CUDA_BRDF_H_

#include "math.cuh"
#include "material.cuh"
#include "random.cuh"
#include "debug.cuh"

namespace kobra {

namespace cuda {

// Smith shadow-masking function (single)
__device__ float G1(float3 n, float3 v, Material mat)
{
	if (dot(v, n) <= 0.0f)
		return 0.0f;

	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, v), 0.0f, 0.999f));

	float tan_theta = tan(theta);

	float denom = 1 + sqrt(1 + alpha * alpha * tan_theta * tan_theta);
	return 2.0f/denom;
}

// Smith shadow-masking function (double)
__device__ float G(float3 n, float3 wi, float3 wo, Material mat)
{
	return G1(n, wo, mat) * G1(n, wi, mat);
}

// Shlicks approximation to the Fresnel reflectance
__device__ float3 shlick_F(float3 wi, float3 h, Material mat)
{
	float k = pow(1 - dot(wi, h), 5);
	return mat.specular + (1 - mat.specular) * k;
}

// Microfacet distribution functions
struct Microfacets {
	__device__ __forceinline__
	static float GGX(float3 n, float3 h, const Material &mat) {
		float alpha = mat.roughness;
		float theta = acos(clamp(dot(n, h), 0.0f, 0.999f));
		
		return (alpha * alpha)
			/ (M_PI * pow(cos(theta), 4)
			* pow(alpha * alpha + tan(theta) * tan(theta), 2.0f));
	}
};

// True fresnel reflectance
__device__ __forceinline__
float FrDielectric(float cosThetaI, float etaI, float etaT)
{
	cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
	// Potentially swap indices of refraction
	bool entering = cosThetaI > 0.f;
	if (!entering) {
		float temp = etaI;
		etaI = etaT;
		etaT = temp;
		cosThetaI = fabs(cosThetaI);
	}

	// Compute _cosThetaT_ using Snell's law
	float sinThetaI = sqrt(fmax((float)0, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;

	// Handle total internal reflection
	if (sinThetaT >= 1)
		return 1;
	float cosThetaT = sqrt(fmax((float)0, 1 - sinThetaT * sinThetaT));
	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
		((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
		((etaI * cosThetaI) + (etaT * cosThetaT));
	return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__device__ __forceinline__
float3 Refract(const float3 &wi, const float3 &n, float eta) {
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1)
	    return reflect(wi, n);
    float cosThetaT = sqrt(1 - sin2ThetaT);
    return normalize(eta * -wi + (eta * cosThetaI - cosThetaT) * n);
}

////////////////////
// Shading models //
////////////////////

// Perfect specular reflection
struct SpecularReflection {
	// TODO: evaluate function: sample, brdf and pdf

	// TODO: tr and tf

	// Evaluate the BRDF
	__device__ __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		float eta_i = entering ? 1.0f : mat.refraction;
		float eta_t = entering ? mat.refraction : 1.0f;

		if (dot(wo, n) < 0) n = -n;
		float3 refl = reflect(-wo, n);
		if (length(refl - wi) > 1e-3f)
			return make_float3(0.0f);

		float cos_theta_i = dot(n, wo);
		float fresnel = FrDielectric(cos_theta_i, eta_i, eta_t);
		return make_float3(fresnel)/abs(dot(n, wi));
	}

	// Evaluate the PDF
	__device__ __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		if (dot(wo, n) < 0) n = -n;
		float3 refl = reflect(-wo, n);
		if (length(refl - wi) > 1e-3f)
			return 0.0f;

		return 1;
	}

	// Sample the BRDF
	__device__ __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		if (dot(wo, n) < 0) n = -n;
		out = Shading::eTransmission;
		return reflect(-wo, n);
	}
};

// Perfect specular transmission
struct SpecularTransmission {
	// TODO: evaluate function: sample, brdf and pdf

	// TODO: tr and tf

	// Evaluate the BRDF
	__device__ __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		if (out & Shading::eTransmission) {
			float eta_i = entering ? 1.0f : mat.refraction;
			float eta_t = entering ? mat.refraction : 1.0f;

			if (dot(wo, n) < 0) n = -n;
			float3 refr = Refract(wo, n, eta_i/eta_t);
			if (length(refr - wi) > 1e-3f)
				return make_float3(0.0f);

			float cos_theta_i = dot(n, wi);
			float fresnel = FrDielectric(cos_theta_i, eta_i, eta_t);

			float k = eta_t/eta_i;
			float tr = (1 - fresnel) * (k * k);
			return make_float3(tr)/abs(cos_theta_i);
		}

		assert(false);
		return make_float3(0.0f);
	}

	// Evaluate the PDF
	__device__ __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		float eta_i = entering ? 1.0f : mat.refraction;
		float eta_t = entering ? mat.refraction : 1.0f;

		if (dot(wo, n) < 0) n = -n;
		float3 refr = Refract(wo, n, eta_i/eta_t);
		if (length(refr - wi) > 1e-3f)
			return 0.0f;

		return 1;
	}

	// Sample the BRDF
	__device__ __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		if (mat.type & eTransmission) {
			// For now, just refract
			out = Shading::eTransmission;
			float eta_i = entering ? 1 : mat.refraction;
			float eta_t = entering ? mat.refraction : 1;
			float3 np = dot(wo, n) < 0 ? -n : n;
			return Refract(wo, np, eta_i/eta_t);
		}

		assert(false);
		return make_float3(0.0f);
	}
};

// Cook-Torrance GGX BRDF
struct GGX {
	// Evaluate the BRDF
	__device__ __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		if (dot(wo, n) <= 0.0f) n = -n;

		float3 h = normalize(wi + wo);

		float3 f = shlick_F(wi, h, mat);
		float g = G(n, wi, wo, mat);
		float d = Microfacets::GGX(n, h, mat);

		float3 num = f * g * d;
		float denom = 4 * dot(wi, n) * dot(wo, n);

		return num / denom;
	}

	// Evaluate the PDF
	__device__ __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		if (dot(wo, n) <= 0.0f) n = -n;

		float3 h = normalize(wi + wo);

		float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
		float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

		float t = 1.0f;
		if (avg_Kd + avg_Ks > 0.0f)
			t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

		float term1 = dot(n, wi)/M_PI;
		float term2 = Microfacets::GGX(n, h, mat) * G(n, wi, wo, mat)
			* dot(wo, h) / (4 * dot(wi, h) * dot(wo, n));
		// float term2 = Microfacets::GGX(n, h, mat) * dot(n, h)/(4.0f * dot(wi, h));

		return (1 - t) * term1 + t * term2;
	}

	// Sample the BRDF
	__device__ __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		if (dot(wo, n) <= 0.0f) n = -n;

		float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
		float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

		float t = 1.0f;
		if (avg_Kd + avg_Ks > 0.0f)
			t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

		float3 eta = fract(random3(seed));
		if (eta.x < t) {
			// Specular sampling
			float k = sqrt(eta.y/(1 - eta.y));
			float theta = atan(k * mat.roughness);
			float phi = 2.0f * M_PI * eta.z;

			float3 h = float3 {
				sin(theta) * cos(phi),
				sin(theta) * sin(phi),
				cos(theta)
			};

			h = rotate(h, n);

			// TODO: change this:
			out = Shading::eDiffuse;
			return reflect(-wo, h);
		}

		// Diffuse sampling
		float theta = acos(sqrt(eta.y));
		float phi = 2.0f * M_PI * eta.z;

		float3 s = float3 {
			sin(theta) * cos(phi),
			sin(theta) * sin(phi),
			cos(theta)
		};

		out = Shading::eDiffuse;
		return rotate(s, n);
	}
};

// brdf, sample, pdf based on shading model

}

}

#endif
