#pragma once

// Engine headers
#include "core.cuh"
#include "debug.cuh"
#include "material.cuh"
#include "math.cuh"
#include "random.cuh"

namespace kobra {

namespace cuda {

// TODO: util header
// True fresnel reflectance
__forceinline__ __device__
float fr_dielectric(float cosThetaI, float etaI, float etaT)
{
	cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
	// Potentially swap indices of refraction
	bool entering = cosThetaI > 0.f;
	if (!entering) {
		float temp = etaI;
		etaI = etaT;
		etaT = temp;
		cosThetaI = fabsf(cosThetaI);
	}

	// Compute _cosThetaT_ using Snell's law
	float sinThetaI = sqrt(fmaxf((float) 0, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;

	// Handle total internal reflection
	if (sinThetaT >= 1)
		return 1;

	float cosThetaT = sqrt(fmax((float) 0, 1 - sinThetaT * sinThetaT));

	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
		((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
		((etaI * cosThetaI) + (etaT * cosThetaT));

	return (Rparl * Rparl + Rperp * Rperp)/2;
}

__forceinline__ __device__
float3 refract(const float3 &wi, const float3 &n, float eta)
{
	float cosThetaI = dot(n, wi);
	float sin2ThetaI = fmax(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	// TODO: which way is correct?
	if (sin2ThetaT >= 1)
		return reflect(wi, n);
		// return float3 {0, 0, 0};

	float cosThetaT = sqrt(1 - sin2ThetaT);
	return normalize(eta * -wi + (eta * cosThetaI - cosThetaT) * n);
}
    
#define PI 3.14159265358979323846f

// Surface hit structure for BSDF calculations
struct SurfaceHit {
	Material	mat;
	bool		entering;
	float3		n;
	float3		wo;
	float3		x;
        // TODO: refactor to backfacing
};

// Orthonormal basis structure
struct ONB {
        float3 u;
        float3 v;
        float3 w;

        __forceinline__ __device__
        float3 local(const float3 &a) const {
                return a.x * u + a.y * v + a.z * w;
        }

        __forceinline__ __device__
        static ONB from_normal(const float3 &n) {
                ONB onb;
                onb.w = normalize(n);

                float3 a = (fabs(onb.w.x) > 0.9) ? float3 { 0, 1, 0 } : float3 { 1, 0, 0 };
                onb.v = normalize(cross(onb.w, a));
                onb.u = cross(onb.w, onb.v);

                return onb;
        }
};

// General scattering structure
struct ScatteringResult {
        float3 brdf = {0, 0, 0};
        float3 wo = {0, 0, 0};
        float pdf = 0;
        // TODO: ray type
};

struct ScatteringBase {
        SurfaceHit sh;
        ONB onb;

        __forceinline__ __device__
        ScatteringBase(const SurfaceHit &sh)
                        : sh(sh), onb(ONB::from_normal(sh.n)) {}
};

// Fresnel glass
struct Glass : ScatteringBase {
        float eta;
        float3 T;

        __forceinline__ __device__
        Glass(const SurfaceHit &sh) : ScatteringBase(sh), eta(sh.mat.refraction) {
                T = sh.mat.specular;
        }

        __forceinline__ __device__
        float pdf(const float3 &wi, float3 &wo) {
                return 0;
        }

        __forceinline__ __device__
        float3 brdf(const float3 &wi, float3 &wo) {
                return float3 { 0, 0, 0 };
        }

        __forceinline__ __device__
        float3 refract(const float3 &v, const float3 &n, float ni_over_nt) {
                float cos_theta = dot(-v, n);
                float3 r_out_parallel = ni_over_nt * (v + cos_theta * n);
                float r_out_length = length(r_out_parallel);
                float r_out_length_squared = r_out_length * r_out_length;
                float3 r_out_perp = -sqrtf(fabs(1.0f - r_out_length_squared)) * n;
                return r_out_parallel + r_out_perp;
        }

        __forceinline__ __device__
        void sample(const float3 &wi, float3 &seed, ScatteringResult &result) {
                float cos_theta = min(dot(-wi, sh.n), 1.0f);
                float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

                bool reflect_ray = (eta * sin_theta > 1.0);

                float ni_over_nt = 1.0/eta;
                // TODO: backfacing
                if (!sh.entering)
                    ni_over_nt = eta;

                float3 wo = reflect_ray ? reflect(wi, sh.n) : refract(wi, sh.n, ni_over_nt);
                // RayType type = eSpecular;
                // if (!reflect_ray)
                //     type = type | eRefractive;

                result.brdf = T;
                result.wo = wo;
                result.pdf = 1.0f;
                // return { Vector3 { 1, 1, 1 }, wo, 1, type };
        }
};

// Smith shadow-masking function (single)
__forceinline__ __device__
float G1(float3 n, float3 v, Material mat)
{
	if (dot(v, n) <= 0.0f)
		return 0.0f;

	float alpha = mat.roughness;
	float theta = acosf(clamp(dot(n, v), 0.0f, 0.999f));

	float tan_theta = __tanf(theta);

	float denom = 1 + sqrtf(1 + alpha * alpha * tan_theta * tan_theta);
	return 2.0f/denom;
}

// Smith shadow-masking function (double)
__forceinline__ __device__
float G(float3 n, float3 wi, float3 wo, Material mat)
{
	return G1(n, wo, mat) * G1(n, wi, mat);
}

// Shlicks approximation to the Fresnel reflectance
__forceinline__ __device__
float3 shlick_F(float3 wi, float3 h, Material mat)
{
	float k = powf(1 - dot(wi, h), 5);
	return mat.specular + (1 - mat.specular) * k;
}

// Microfacet distribution functions
struct Microfacets {
	__forceinline__ __device__
	static float GGX(float3 n, float3 h, const Material &mat) {
		float alpha = mat.roughness;
		float theta = acosf(clamp(dot(n, h), 0.0f, 1.0f));

		return (alpha * alpha)
			/ (PI * powf(__cosf(theta), 4)
			* powf(alpha * alpha + __tanf(theta) * __tanf(theta), 2.0f));
	}
};

////////////////////
// Shading models //
////////////////////

// Perfect specular reflection
struct SpecularReflection {
	// Evaluate the BRDF
	__forceinline__ __device__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering,
			Shading out, bool isrec = false)
	{
		return {0, 0, 0};
	}

	// Evaluate the PDF
	__forceinline__ __device__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		return 0.0f;
	}

	// Sample the BRDF
	__forceinline__ __device__
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
	__device__ __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		return { 0, 0, 0 };
	}

	// Evaluate the PDF
	__device__ __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		return 0.0f;
	}

	// Sample the BRDF
        // TODO: return the brdf and pdf as well
	__device__ __forceinline__
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
	__device__ __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out, bool isrec = false)
	{
		return {0, 0, 0};
	}

	// Evaluate the PDF
	__device__ __forceinline__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		return 0.0f;
	}

	// Sample the BRDF
	__device__ __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		float eta_i = entering ? 1 : mat.refraction;
		float eta_t = entering ? mat.refraction : 1;

		float F = fr_dielectric(dot(wo, n), eta_i, eta_t);

		float eta = rand_uniform(seed);
		if (eta < F) {
			out = Shading::eReflection;
			// return {1, 0, 0};
			return reflect(-wo, n);
		} else {
			out = Shading::eTransmission;
			// return {0, 0, 1};
			return refract(wo, n, 1/mat.refraction);
		}
	}
};

// Cook-Torrance GGX BRDF
struct GGX {
	// Evaluate the BRDF
	__forceinline__ __device__
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

		return num / (denom + 1e-6f);
	}

	// Evaluate the PDF
	__forceinline__ __device__
	static float pdf(const Material &mat, float3 n, float3 wi,
			float3 wo, bool entering, Shading out)
	{
		if (dot(wo, n) <= 0.0f) n = -n;
		if (dot(wi, n) <= 0.0f)
			return 0.0f;

		float3 h = normalize(wi + wo);

		float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
		float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

		float t = 1.0f;
		if (avg_Kd + avg_Ks > 0.0f)
			t = fmax(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

		float term1 = dot(n, wi)/PI;
		// float term2 = Microfacets::GGX(n, h, mat) * G(n, wi, wo, mat)
		//	* dot(wo, h) / (4 * dot(wi, h) * dot(wo, n));

		float term2 = Microfacets::GGX(n, h, mat)
			* dot(n, h)/(4.0f * dot(wi, h));

		float pdf = (1 - t) * term1 + t * term2;
		return pdf;
	}

	// Sample the BRDF
	__forceinline__ __device__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			bool entering, float3 &seed, Shading &out)
	{
		if (dot(wo, n) <= 0.0f) n = -n;

		float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
		float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

		float t = 1.0f;
		// if (avg_Kd + avg_Ks > 0.0f)
		//	t = fmax(avg_Ks/(avg_Kd + avg_Ks), 0.25f);
		if (avg_Kd + avg_Ks > 0.0f)
			t = avg_Ks/(avg_Kd + avg_Ks);

		float3 eta = rand_uniform_3f(seed);
		if (eta.x < t) {
			// Specular sampling
			float k = sqrtf(eta.y/(1 - eta.y));
			float theta = atanf(k * mat.roughness);
			float phi = 2.0f * PI * eta.z;

			float3 h = float3 {
				__sinf(theta) * __cosf(phi),
				__sinf(theta) * __sinf(phi),
				__cosf(theta)
			};

			h = rotate(h, n);

			// TODO: change this:
			out = Shading::eDiffuse;
			float3 wi = reflect(-wo, h);
			return wi;
		}

		// Diffuse sampling
		float theta = acosf(sqrtf(eta.y));
		float phi = 2.0f * PI * eta.z;

		float3 s = float3 {
			__sinf(theta) * __cosf(phi),
			__sinf(theta) * __sinf(phi),
			__cosf(theta)
		};

		out = Shading::eDiffuse;
		return rotate(s, n);
	}
};

// Evaluate BRDF of material
__forceinline__ __device__
float3 brdf(const SurfaceHit &sh, float3 wi, Shading out)
{
	// TODO: diffuse should be in conjunction with the material
	// TODO: plus specular lobe in either case...
	// TODO: Implement PBRT specular transmission with Tr...
	// also fresnel transmission and microfacet transmission
	if (out & Shading::eTransmission)
		return sh.mat.diffuse/PI + Glass(sh).brdf(sh.wo, wi);
		// return sh.mat.diffuse/PI + FresnelSpecular::brdf(sh.mat, sh.n, wi, sh.wo, sh.entering, out);

	return sh.mat.diffuse/PI + GGX::brdf(sh.mat, sh.n, wi, sh.wo, sh.entering, out);
}

// Evaluate PDF of BRDF
__forceinline__ __device__
float pdf(const SurfaceHit &sh, float3 wi, Shading out)
{
	if (out & Shading::eTransmission)
                return Glass(sh).pdf(sh.wo, wi);
		// return FresnelSpecular::pdf(sh.mat, sh.n, wi, sh.wo, sh.entering, out);

	return GGX::pdf(sh.mat, sh.n, wi, sh.wo, sh.entering, out);
}

// Sample BRDF
__forceinline__ __device__
float3 sample(const SurfaceHit &sh, Shading &out, Seed seed)
{
	if (sh.mat.type & Shading::eTransmission) {
                ScatteringResult result;
                Glass(sh).sample(sh.wo, seed, result);
                return result.wo;
        }
		// return FresnelSpecular::sample(sh.mat, sh.n, sh.wo, sh.entering, seed, out);

	return GGX::sample(sh.mat, sh.n, sh.wo, sh.entering, seed, out);
}

// Evaluate BRDF: sample, brdf, pdf
template <class BxDF>
__device__ __forceinline__
float3 eval(const SurfaceHit &sh, float3 &wi, float &in_pdf, Shading &out, Seed seed)
{
	// TODO: pack ags into struct
	wi = sample(sh, out, seed);
	// wi = sample(mat, n, wo, entering, seed, out);
	if (length(wi) < 1e-6f)
		return make_float3(0.0f);

	in_pdf = pdf(sh, wi, out);
	return brdf(sh, wi, out);

	// in_pdf = pdf(mat, n, wi, wo, entering, out);
	// return brdf(mat, n, wi, wo, entering, out);
}

template <>
__device__ __forceinline__
float3 eval <SpecularTransmission>
(const SurfaceHit &sh, float3 &wi, float &in_pdf, Shading &out, Seed seed)
{
        ScatteringResult result;
        Glass(sh).sample(sh.wo, seed, result);
        wi = result.wo;
        in_pdf = result.pdf;
        out = sh.mat.type;
        return result.brdf;

	// out = Shading::eTransmission;
	// float eta_i = sh.entering ? 1 : sh.mat.refraction;
	// float eta_t = sh.entering ? sh.mat.refraction : 1;
	//
	// float3 n = sh.n;
	// if (dot(n, sh.wo) < 0)
	// 	n = -n;
	//
	// float eta = eta_i/eta_t;
	// wi = refract(sh.wo, n, eta);
	// if (length(wi) < 1e-6f)
	// 	return make_float3(0);
	// in_pdf = 1;
	//
	// float fr = fr_dielectric(dot(n, wi), eta_i, eta_t);
	// return make_float3(1 - fr) * (eta * eta)/fabsf(dot(n, wi));
}

template <>
__device__ __forceinline__
float3 eval <SpecularReflection>
(const SurfaceHit &sh, float3 &wi, float &in_pdf, Shading &out, Seed seed)
{
	float eta_i = sh.entering ? 1 : sh.mat.refraction;
	float eta_t = sh.entering ? sh.mat.refraction : 1;

	float3 n = sh.n;
	if (dot(n, sh.wo) < 0)
		n = -n;

	wi = reflect(-sh.wo, n);
	in_pdf = 1;

	float fr = fr_dielectric(dot(n, wi), eta_i, eta_t);
	return make_float3(fr)/fabsf(dot(n, wi));
}

template <>
__forceinline__ __device__ 
float3 eval <FresnelSpecular>
(const SurfaceHit &sh, float3 &wi, float &in_pdf, Shading &out, Seed seed)
{
	float eta_i = sh.entering ? 1 : sh.mat.refraction;
	float eta_t = sh.entering ? sh.mat.refraction : 1;

	float F = fr_dielectric(dot(sh.wo, sh.n), eta_i, eta_t);

	float eta = rand_uniform(seed);
	if (eta < F) {
		wi = reflect(-sh.wo, sh.n);
		in_pdf = F;
		// float fr = kobra::cuda::fr_dielectric(dot(n, wi), eta_i, eta_t);
		return make_float3(F)/abs(dot(sh.n, wi));
	} else {
		out = Shading::eTransmission;
		float eta = eta_i/eta_t;
		wi = refract(sh.wo, sh.n, eta);
		if (length(wi) < 1e-6f)
			return make_float3(0);

		in_pdf = 1 - F;
		// float fr = kobra::cuda::fr_dielectric(dot(n, wi), eta_i, eta_t);
		return make_float3(1 - F) * (eta * eta)/abs(dot(sh.n, wi));
	}
}

// TODO: union in material for different shading models
__forceinline__ __device__ 
float3 eval(const SurfaceHit &sh, float3 &wi, float &in_pdf, Shading &out, Seed seed)
{
	if (sh.mat.type & Shading::eTransmission) {
                ScatteringResult result;
                Glass(sh).sample(sh.wo, seed, result);
                wi = result.wo;
                in_pdf = result.pdf;
                out = sh.mat.type;
                return result.brdf;
        }

		// return eval <FresnelSpecular> (sh, wi, in_pdf, out, seed);

	// Fallback to GGX
	return eval <GGX> (sh, wi, in_pdf, out, seed);
}

}

}
