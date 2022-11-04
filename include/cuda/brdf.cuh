#ifndef KOBRA_CUDA_BRDF_H_
#define KOBRA_CUDA_BRDF_H_

// Engine headers
#include "core.cuh"
#include "debug.cuh"
#include "material.cuh"
#include "math.cuh"
#include "random.cuh"

namespace kobra {

namespace cuda {

// Surface hit structure for BSDF calculations
struct SurfaceHit {
	float3 x;
	float3 wo;
	float3 n;
	Material mat;
	bool entering;
};

// Smith shadow-masking function (single)
KCUDA_INLINE KCUDA_HOST_DEVICE
float G1(float3 n, float3 v, Material mat)
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
KCUDA_INLINE KCUDA_HOST_DEVICE
float G(float3 n, float3 wi, float3 wo, Material mat)
{
	return G1(n, wo, mat) * G1(n, wi, mat);
}

// Shlicks approximation to the Fresnel reflectance
KCUDA_INLINE KCUDA_HOST_DEVICE
float3 shlick_F(float3 wi, float3 h, Material mat)
{
	float k = pow(1 - dot(wi, h), 5);
	return mat.specular + (1 - mat.specular) * k;
}

// Microfacet distribution functions
struct Microfacets {
	KCUDA_INLINE KCUDA_HOST_DEVICE
	static float GGX(float3 n, float3 h, const Material &mat) {
		float alpha = mat.roughness;
		float theta = acos(clamp(dot(n, h), 0.0f, 0.999f));
		
		return (alpha * alpha)
			/ (M_PI * pow(cos(theta), 4)
			* pow(alpha * alpha + tan(theta) * tan(theta), 2.0f));
	}
};

// True fresnel reflectance
KCUDA_INLINE KCUDA_HOST_DEVICE
float fr_dielectric(float cosThetaI, float etaI, float etaT)
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

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 refract(const float3 &wi, const float3 &n, float eta)
{
	float cosThetaI = dot(n, wi);
	float sin2ThetaI = fmax(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1)
		return reflect(wi, n);
		// return float3 {0, 0, 0};

	float cosThetaT = sqrt(1 - sin2ThetaT);
	return normalize(eta * -wi + (eta * cosThetaI - cosThetaT) * n);
}

////////////////////
// Shading models //
////////////////////

// Perfect specular reflection
struct SpecularReflection {
	// Evaluate the BRDF
	KCUDA_INLINE KCUDA_HOST_DEVICE
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering,
			Shading out, bool isrec = false)
	{
		return {0, 0, 0};
	}

	// Evaluate the PDF
	KCUDA_INLINE KCUDA_HOST_DEVICE
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		return 0.0f;
	}

	// Sample the BRDF
	KCUDA_INLINE KCUDA_HOST_DEVICE
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
	// TODO: tr and tf

	// Evaluate the BRDF
	KCUDA_HOST_DEVICE KCUDA_INLINE
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		return {0, 0, 0};
	}

	// Evaluate the PDF
	KCUDA_HOST_DEVICE __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		return 0.0f;
	}

	// Sample the BRDF
	KCUDA_HOST_DEVICE __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		// For now, just refract
		out = Shading::eTransmission;
		float eta_i = entering ? 1 : mat.refraction;
		float eta_t = entering ? mat.refraction : 1;
		float3 np = dot(wo, n) < 0 ? -n : n;

		// WARNING: can return 0 vector...
		return refract(wo, np, eta_i/eta_t);
	}
};

// Fresnel modulated BRDF
struct FresnelSpecular {
	// Evaluate the BRDF
	KCUDA_HOST_DEVICE __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		return {0, 0, 0};
	}

	// Evaluate the PDF
	KCUDA_HOST_DEVICE __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		return 0.0f;
	}

	// Sample the BRDF
	KCUDA_HOST_DEVICE __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		float eta_i = entering ? 1 : mat.refraction;
		float eta_t = entering ? mat.refraction : 1;

		float F = fr_dielectric(dot(wo, n), eta_i, eta_t);

		float eta = rand_uniform(seed);
		if (eta < F) {
			out = Shading::eReflection;
			return {1, 0, 0};
			return reflect(-wo, n);
		} else {
			out = Shading::eTransmission;
			return {0, 0, 1};
			return refract(wo, n, 1/mat.refraction);
		}
	}
};

// Cook-Torrance GGX BRDF
struct GGX {
	// Evaluate the BRDF
	KCUDA_INLINE KCUDA_HOST_DEVICE
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
	KCUDA_INLINE KCUDA_HOST_DEVICE
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		if (dot(wo, n) <= 0.0f) n = -n;

		float3 h = normalize(wi + wo);

		float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
		float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

		float t = 1.0f;
		if (avg_Kd + avg_Ks > 0.0f)
			t = fmax(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

		float term1 = dot(n, wi)/M_PI;
		float term2 = Microfacets::GGX(n, h, mat) * G(n, wi, wo, mat)
			* dot(wo, h) / (4 * dot(wi, h) * dot(wo, n));
		// float term2 = Microfacets::GGX(n, h, mat) * dot(n, h)/(4.0f * dot(wi, h));

		return (1 - t) * term1 + t * term2;
	}

	// Sample the BRDF
	KCUDA_INLINE KCUDA_HOST_DEVICE
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		if (dot(wo, n) <= 0.0f) n = -n;

		float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
		float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

		float t = 1.0f;
		if (avg_Kd + avg_Ks > 0.0f)
			t = fmax(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

		float3 eta = rand_uniform_3f(seed);
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

// Evaluate BRDF of material
KCUDA_INLINE KCUDA_HOST_DEVICE
float3 brdf(const Material &mat, float3 n, float3 wi,
		float3 wo, bool entering, Shading out)
{
	// TODO: diffuse should be in conjunction with the material
	if (out & Shading::eTransmission)
		return SpecularTransmission::brdf(mat, n, wi, wo, entering, out);
	
	return mat.diffuse/M_PI + GGX::brdf(mat, n, wi, wo, entering, out);
}

// Evaluate PDF of BRDF
KCUDA_INLINE KCUDA_HOST_DEVICE
float pdf(const Material &mat, float3 n, float3 wi,
		float3 wo, bool entering, Shading out)
{
	if (out & Shading::eTransmission)
		return SpecularTransmission::pdf(mat, n, wi, wo, entering, out);
	
	return GGX::pdf(mat, n, wi, wo, entering, out);
}

// Sample BRDF
KCUDA_INLINE KCUDA_HOST_DEVICE
float3 sample(const Material &mat, float3 n, float3 wo,
		bool entering, float3 &seed, Shading &out)
{
	if (mat.type & Shading::eTransmission)
		return SpecularTransmission::sample(mat, n, wo, entering, seed, out);

	return GGX::sample(mat, n, wo, entering, seed, out);
}

// Evaluate BRDF: sample, brdf, pdf
template <class BxDF>
KCUDA_HOST_DEVICE __forceinline__
float3 eval
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &in_pdf, Shading &out, float3 &seed)
{
	// TODO: pack ags into struct
	wi = sample(mat, n, wo, entering, seed, out);
	if (length(wi) < 1e-6f)
	 	return make_float3(0.0f);

	in_pdf = pdf(mat, n, wi, wo, entering, out);
	return brdf(mat, n, wi, wo, entering, out);
}

template <>
KCUDA_HOST_DEVICE __forceinline__
float3 eval <SpecularTransmission>
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	out = Shading::eTransmission;
	float eta_i = entering ? 1 : mat.refraction;
	float eta_t = entering ? mat.refraction : 1;

	if (dot(n, wo) < 0)
		n = -n;

	float eta = eta_i/eta_t;
	wi = refract(wo, n, eta);
	if (length(wi) < 1e-6f)
		return make_float3(0);
	pdf = 1;

	float fr = fr_dielectric(dot(n, wi), eta_i, eta_t);
	return make_float3(1 - fr) * (eta * eta)/abs(dot(n, wi));
}

template <>
KCUDA_HOST_DEVICE __forceinline__
float3 eval <SpecularReflection>
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	float eta_i = entering ? 1 : mat.refraction;
	float eta_t = entering ? mat.refraction : 1;

	if (dot(n, wo) < 0)
		n = -n;

	wi = reflect(-wo, n);
	pdf = 1;

	float fr = fr_dielectric(dot(n, wi), eta_i, eta_t);
	return make_float3(fr)/abs(dot(n, wi));
}

template <>
KCUDA_HOST_DEVICE __forceinline__
float3 eval <FresnelSpecular>
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	float eta_i = entering ? 1 : mat.refraction;
	float eta_t = entering ? mat.refraction : 1;

	float F = fr_dielectric(dot(wo, n), eta_i, eta_t);

	float eta = rand_uniform(seed);
	if (eta < F) {
		wi = reflect(-wo, n);
		pdf = F;
		// float fr = kobra::cuda::fr_dielectric(dot(n, wi), eta_i, eta_t);
		return make_float3(F)/abs(dot(n, wi));
	} else {
		out = Shading::eTransmission;
		float eta = eta_i/eta_t;
		wi = refract(wo, n, eta);
		if (length(wi) < 1e-6f)
			return make_float3(0);

		pdf = 1 - F;
		// float fr = kobra::cuda::fr_dielectric(dot(n, wi), eta_i, eta_t);
		return make_float3(1 - F) * (eta * eta)/abs(dot(n, wi));
	}
}

// TOdo: union in material for different shading models
KCUDA_HOST_DEVICE __forceinline__
float3 eval(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	if (mat.type & Shading::eTransmission)
		return eval <SpecularTransmission> (mat, n, wo, entering, wi, pdf, out, seed);

	// Fallback to GGX
	return eval <GGX> (mat, n, wo, entering, wi, pdf, out, seed);
}

}

}

#endif
