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
float Fresnel(float cos_theta_i, float eta_i, float eta_t)
{
	cos_theta_i = clamp(cos_theta_i, -1.0f, 1.0f);
	
	if (cos_theta_i > 0.0f) {
		float temp = eta_i;
		eta_i = eta_t;
		eta_t = temp;
	} else {
		cos_theta_i = -cos_theta_i;
	}

	float sin_theta_i = sqrt(max(1 - cos_theta_i * cos_theta_i, 0.0f));
	float sin_theta_t = (eta_i/eta_t) * sin_theta_i;
	if (sin_theta_t >= 1.0f)
		return 1.0f;

	float cos_theta_t = sqrt(max(1 - sin_theta_t * sin_theta_t, 0.0f));

	float r_parallel = (eta_t * cos_theta_i - eta_i * cos_theta_t)
		/ (eta_t * cos_theta_i + eta_i * cos_theta_t);
	float r_perpendicular = (eta_i * cos_theta_i - eta_t * cos_theta_t)
		/ (eta_i * cos_theta_i + eta_t * cos_theta_t);

	return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2.0f;
}

// Shading models
struct GGX {
	// Evaluate the BRDF
	__device__ __forceinline__
	static float3 brdf(const Material &mat, float3 n, float3 wi,
			float3 wo, float ior, Shading out, bool isrec = false)
	{
		if (out & Shading::eTransmission) {
			float inv_eta = mat.refraction/ior;
			float3 refr = refract(wo, n, 1/inv_eta);
			if (length(refr - wi) > 1e-3f)
				return make_float3(0.0f);

			float cos_theta_i = -dot(wi, n);
			float fresnel = Fresnel(cos_theta_i, ior, mat.refraction);
			float tr = (1 - fresnel) * (inv_eta * inv_eta)/abs(cos_theta_i);
			return make_float3(tr);

			/* print("GGX transmission!\n");
			float cos_theta_i = dot(n, wo);
			assert(!isnan(cos_theta_i));
			
			float3 refr = refract(wo, n, 1/eta.x);
			if (length(refr - wi) > 1e-3f) {
				print("Fake refraction: %f %f %f\n", refr.x, refr.y, refr.z);
				print("\twi = %f %f %f\n", wi.x, wi.y, wi.z);

				/* if (isrec)
					assert(false);
				return make_float3(0, 0, 0);
			}

			print("Cos theta i = %f\n", cos_theta_i);

			float Fr = Fresnel(cos_theta_i, ior, mat.roughness);
			assert(!isnan(Fr));
			print("Fresnel = %f\n", Fr);
			return make_float3(Fr);
			float3 o = (eta * eta) * (1 - Fr)/dot(wi, n);
			assert(!isnan(o.x) && !isnan(o.y) && !isnan(o.z));
			return o; */
		}
		
		print("GGX specular!\n");

		if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
			return float3 {0.0f, 0.0f, 0.0f};

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
			float3 wo, Shading out)
	{
		if (out & Shading::eTransmission)
			print("Sampled transmission\n");
		else
			print("Sampled reflection\n");

		// TODO: refactor shading type to ray type
		if (out & Shading::eTransmission) {
			return 1;
		}

		if (dot(wi, n) <= 0.0f || dot(wo, n) < 0.0f)
			return 0.0f;

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
			float ior, float3 &seed, Shading &out)
	{
		if (mat.type & eTransmission) {
			// TODO: check if both refraction and reflection are
			// enabled and sample accordingly

			// For now, just refract
			out = Shading::eTransmission;
			return refract(wo, n, ior/mat.refraction);
		}

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
