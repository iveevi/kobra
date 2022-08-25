#ifndef KOBRA_SHADERS_BRDF_H_
#define KOBRA_SHADERS_BRDF_H_

// Shader modules
#include "constants.glsl"
#include "material.glsl"
#include "math.glsl"
#include "random.glsl"

// TODO: move GGX specific things to ggx.glsl

// GGX microfacet distribution function
float ggx_d(Material mat, vec3 n, vec3 h)
{
	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, h), 0, 0.999f));
	return (alpha * alpha)
		/ (PI * pow(cos(theta), 4)
		* pow(alpha * alpha + tan(theta) * tan(theta), 2.0f));
}

// Smith shadow-masking function (single)
float G1(Material mat, vec3 n, vec3 v)
{
	if (dot(v, n) <= 0.0f)
		return 0.0f;

	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, v), 0, 0.999f));

	float tan_theta = tan(theta);

	float denom = 1 + sqrt(1 + alpha * alpha * tan_theta * tan_theta);
	return 2.0f/denom;
}

// Smith shadow-masking function (double)
float G(Material mat, vec3 n, vec3 wi, vec3 wo)
{
	return G1(mat, n, wo) * G1(mat, n, wi);
}

// Shlicks approximation to the Fresnel reflectance
vec3 ggx_f(Material mat, vec3 wi, vec3 h)
{
	float k = pow(1 - dot(wi, h), 5);
	return mat.specular + (1 - mat.specular) * k;
}

// GGX specular brdf
vec3 ggx_brdf(Material mat, vec3 n, vec3 wi, vec3 wo)
{
	if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
		return vec3(0.0f);

	vec3 h = normalize(wi + wo);

	vec3 f = ggx_f(mat, wi, h);
	float g = G(mat, n, wi, wo);
	float d = ggx_d(mat, n, h);

	vec3 num = f * g * d;
	float denom = 4 * dot(wi, n) * dot(wo, n);

	return num / denom;
}

// GGX pdf
float ggx_pdf(Material mat, vec3 n, vec3 wi, vec3 wo)
{
	if (dot(wi, n) <= 0.0f || dot(wo, n) < 0.0f)
		return 0.0f;

	vec3 h = normalize(wi + wo);

	float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
	float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

	float t = 1.0f;
	if (avg_Kd + avg_Ks > 0.0f)
		t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

	float term1 = dot(n, wi)/PI;
	float term2 = ggx_d(mat, n, h) * dot(n, h)/(4.0f * dot(wi, h));

	return (1 - t) * term1 + t * term2;
}

vec3 ggx_sample(Material mat, vec3 n, vec3 wo, inout vec3 seed)
{
	float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
	float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

	float t = 1.0f;
	if (avg_Kd + avg_Ks > 0.0f)
		t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

	vec3 eta = fract(random3(seed));

	if (eta.x < t) {
		// Specular sampling
		float k = sqrt(eta.y/(1 - eta.y));
		float theta = atan(k * mat.roughness);
		float phi = 2.0f * PI * eta.z;

		vec3 h = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
		h = rotate(h, n);

		return reflect(-wo, h);
	}

	// Diffuse sampling
	float theta = acos(sqrt(eta.y));
	float phi = 2.0f * PI * eta.z;

	vec3 s = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
	return rotate(s, n);
}

#endif
