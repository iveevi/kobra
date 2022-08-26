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
			if (dot(wo, n) * dot(wi, n) >= 0.0f)
				return make_float3(0.0f);

			float eta = ior/mat.refraction;

			float3 h = normalize(wo - wi * eta);

			float d_wo_h = dot(wo, h);
			float d_wi_h = dot(-wi, h);

			float fr = Fresnel(d_wo_h, ior, mat.refraction);
			float d = Microfacets::GGX(n, h, mat);
			float g = G(n, -wi, wo, mat);

			float num = d * g * (1 - fr);
			float denom = (d_wo_h + eta * d_wi_h);
			denom *= denom;

			float cos_term = (d_wo_h * d_wi_h)/(dot(wo, n) * dot(-wi, n));

			return make_float3(abs(cos_term * num/denom));
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
			float3 wo, float ior, Shading out)
	{
		if (out & Shading::eTransmission)
			print("PDF - Sampled transmission\n");
		else
			print("PDF - Sampled reflection\n");

		// TODO: refactor shading type to ray type
		if (out & Shading::eTransmission) {
			if (dot(wo, n) * dot(wi, n) >= 0.0f)
				return 0.0f;

			float eta = ior/mat.refraction;
			float3 h = normalize(wo - wi * eta);

			float sqrt_denom = dot(wo, h) + eta * dot(-wi, h);
			float dh_dwo = (eta * eta * dot(-wi, h)) / (sqrt_denom * sqrt_denom);

			float D = Microfacets::GGX(n, h, mat)
			 	* G1(n, wo, mat) * abs(dot(wo, h))/abs(dot(wo, n));
			return abs(D * dh_dwo);
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
		float term2 = Microfacets::GGX(n, h, mat) * G1(n, wo, mat)
			* dot(wo, h) / (4 * dot(wi, h) * dot(wo, n));

		return (1 - t) * term1 + t * term2;
	}

	// Sample the BRDF
	__device__ __forceinline__
	static float3 sample(const Material &mat, float3 n, float3 wo,
			float ior, float3 &seed, Shading &out)
	{
		if (mat.type & eTransmission) {
			print("Sample - Sampling transmission!\n");
			out = Shading::eTransmission;

			// Sample half vector
			float3 eta = fract(random3(seed));
			float k = sqrt(eta.y/(1 - eta.y));

			float theta = atan(k * mat.roughness);
			float phi = 2.0f * M_PI * eta.z;

			float3 h = float3 {
				sin(theta) * cos(phi),
				sin(theta) * sin(phi),
				cos(theta)
			};

			h = rotate(h, n);

			print("\tSampled half vector: %f, %f, %f\n",
				h.x, h.y, h.z);
			print("\two = %f, %f, %f\n", wo.x, wo.y, wo.z);

			// Return refracted ray
			float3 o = normalize(refract(wo, h, ior/mat.refraction));
			print("\tSampled ray: %f, %f, %f\n", o.x, o.y, o.z);
			print("\tn = %f, %f, %f\n", n.x, n.y, n.z);
			print("\two * h = %f\n", dot(wo, h));
			print("\two * n = %f\n", dot(wo, n));
			return o;
		}

		print("Sample - Sampling reflection!\n");

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
