#version 450

// Import all modules and headers
#include "../../include/types.hpp"
#include "bindings.h"
#include "modules/layouts.glsl"
#include "modules/ray.glsl"
#include "modules/bbox.glsl"
#include "modules/color.glsl"
#include "modules/random.glsl"
#include "modules/primitives.glsl"
#include "modules/intersect.glsl"
#include "modules/environment.glsl"
#include "modules/bvh.glsl"

// Maximum ray depth
#define MAX_DEPTH 10

const float eps = 1e-3f;
const float p = 2.0f;

// GGX microfacet distribution function
float ggx_d(vec3 n, vec3 h, Material mat)
{
	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, h), 0, 1));
	if (dot(n, h) >= 1.0f)
		theta = 0.0f;

	return (alpha * alpha)
		/ (PI * pow(theta, 4)
		* pow(alpha * alpha + tan(theta) * tan(theta), 2.0f));
}

// Smith shadow-masking function (single)
float G1(vec3 n, vec3 v, Material mat)
{
	if (dot(v, n) <= 0.0f)
		return 0.0f;

	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, v), 0, 1));
	if (dot(n, v) >= 1.0f)
		theta = 0.0f;

	float tan_theta = tan(theta);

	float denom = 1 + sqrt(1 + alpha * alpha * tan_theta * tan_theta);
	return 2.0f/denom;
}

// Smith shadow-masking function (double)
float G(vec3 n, vec3 wi, vec3 wo, Material mat)
{
	return G1(n, wo, mat) * G1(n, wi, mat);
}

// Shlicks approximation to the Fresnel reflectance
vec3 ggx_f(vec3 wi, vec3 h, Material mat)
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

	vec3 f = ggx_f(wi, h, mat);
	float g = G(n, wi, wo, mat);
	float d = ggx_d(n, h, mat);

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
	float term2 = ggx_d(n, h, mat) * dot(n, h)/(4.0f * dot(wi, h));

	return (1 - t) * term1 + t * term2;
}

// Sample from GGX distribution
vec3 rotate(vec3 s, vec3 n)
{
	vec3 w = n;
	vec3 a = vec3(0.0f, 1.0f, 0.0f);

	if (dot(w, a) > 0.999f)
		a = vec3(0.0f, 0.0f, 1.0f);

	vec3 u = normalize(cross(w, a));
	vec3 v = normalize(cross(w, u));

	return u * s.x + v * s.y + w * s.z;
}

vec3 ggx_sample(vec3 n, vec3 wo, Material mat)
{
	float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
	float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

	float t = 1.0f;
	if (avg_Kd + avg_Ks > 0.0f)
		t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

	vec3 eta = fract(random_sphere());

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

// BRDF of material
vec3 brdf(Material mat, vec3 n, vec3 wi, vec3 wo)
{
	return ggx_brdf(mat, n, wi, wo) + mat.diffuse/PI;
}

// Power heurestic
float power(float pdf_f, float pdf_g)
{
	float f = pow(pdf_f, p);
	float g = pow(pdf_g, p);

	return f/(f + g);
}

// Area light methods
vec3 normal(AreaLight light)
{
	// TODO: precompute this
	return normalize(cross(light.ab, light.ac));
}

float area(AreaLight light)
{
	// TODO: precompute this
	return length(cross(light.ab, light.ac));
}

vec3 sample_area_light(AreaLight light)
{
	vec3 rand = random_sphere();
	float u = fract(rand.x);
	float v = fract(rand.y);
	return light.a + u * light.ab + v * light.ac;
}

// Direct lighting
vec3 Ld(vec3 x, vec3 wo, vec3 n, Material mat)
{
	if (mat.type == SHADING_EMISSIVE)
		return vec3(0.0f);

	vec3 contr_nee = vec3(0.0f);
	vec3 contr_brdf = vec3(0.0f);

	// Random area light for NEE
	uvec3 seed = floatBitsToUint(vec3(pc.time, x.x, wo.y));
	uint i = randuint(seed, area_lights.count);
	int light_id = int(i + pc.triangles);
	AreaLight light = area_lights.data[i];

	// NEE
	vec3 lpos = sample_area_light(light);
	vec3 wi = normalize(lpos - x);
	float R = length(lpos - x);

	vec3 f = brdf(mat, n, wi, wo) * dot(n, wi);

	float ldot = abs(dot(normal(light), wi));
	if (ldot > 1e-6) {
		float pdf_light = (R * R)/(area(light) * ldot);
		float pdf_brdf = ggx_pdf(mat, n, wi, wo);

		Ray ray_light = Ray(x, wi, 0, 0);
		Hit hit_light = trace(ray_light);

		if (pdf_light > 1e-9 && hit_light.id == light_id) {
			float weight = power(pdf_light, pdf_brdf);
			weight = 1.0f;

			vec3 intensity = light.color * light.power;
			contr_nee += weight * f * intensity/pdf_light;
		}
	}

	// BRDF
	wi = ggx_sample(n, wo, mat);
	if (dot(wi, n) <= 0.0f)
		return contr_nee;

	f = brdf(mat, n, wi, wo) * dot(n, wi);

	float pdf_brdf = ggx_pdf(mat, n, wi, wo);
	float pdf_light = 0.0f;

	Ray ray_light = Ray(x, wi, 0, 0);
	Intersection it = intersection_light(ray_light, light);

	if (it.time != -1) {
		float R = it.time;
		pdf_light = (R * R)/(area(light) * abs(dot(normal(light), wi)));
	}

	if (pdf_light > 1e-9 && trace(ray_light).id == light_id) {
		float weight = power(pdf_brdf, pdf_light);
		vec3 intensity = light.color * light.power;
		contr_brdf += weight * f * intensity/pdf_brdf;
	}

	return contr_nee + contr_brdf;
}

// Indirect lighting
vec3 Lo(vec3 x, vec3 wo, vec3 n, Material mat, int depth)
{
	vec3 contr = vec3(0.0f);
	vec3 throughput = vec3(1.0f);

	for (int i = 0; i < depth; i++) {
		// Fix normal
		n = normalize(n);
		if (dot(n, wo) <= 0.0f)
			n = -n;

		// Get direct lighting
		vec3 Ld = Ld(x, wo, n, mat);
		contr += throughput * Ld;

		// Generate random direction
		vec3 wi = ggx_sample(n, wo, mat);
		float pdf = ggx_pdf(mat, n, wi, wo);

		if (pdf < 1e-9)
			break;

		vec3 f = brdf(mat, n, wi, wo) * abs(dot(n, wi));
		vec3 T = f/pdf;

		// Get next point
		Ray ray = Ray(x, wi, 0, 0);
		Hit hit = trace(ray);

		if (hit.id == -1) {
			// TODO: should use diffuse instead
			contr += throughput * T * hit.mat.diffuse;
			break;
		}

		// Russian roulette
		float tmax = max(throughput.x, max(throughput.y, throughput.z));
		float q = 1 - min(tmax, 1.0f);
		float r = random();

		if (r < q)
			break;

		// Setup next point
		x = hit.point + hit.normal * 1e-3f;
		n = hit.normal;
		mat = hit.mat;
		throughput *= T/(1 - q);
		wo = -wi;
	}

	return contr;
}

vec3 pathtracer(Ray ray, int depth)
{
	Hit hit = trace(ray);
	if (hit.id == -1)
		return hit.mat.diffuse;

	// Get point and direction
	vec3 x = hit.point + hit.normal * eps;
	vec3 wo = -ray.direction;

	vec3 color = hit.mat.emission + Lo(x, wo, hit.normal, hit.mat, 5);
	return clamp(color, 0.0f, 1.0f);
}

void main()
{
	// Offset from space origin
	uint y0 = pc.skip * gl_WorkGroupID.y + pc.xoffset;
	uint x0 = pc.skip * gl_WorkGroupID.x + pc.yoffset;

	// Return if out of bounds
	if (y0 >= pc.height || x0 >= pc.width)
		return;

	// Get index
	uint index = y0 * pc.width + x0;

	// Set seed
	float rx = fract(sin(x0 * 12 + y0) * PHI);
	float ry = fract(sin(y0 * 98 + x0));

	// Initialiize the random seed
	random_seed = vec3(rx, ry, fract((rx + ry)/pc.time));
	vec2 dimensions = vec2(pc.width, pc.height);

	// Create the ray
	vec2 pixel = vec2(x0, y0) + random_sphere().xy/2.0;
	vec2 uv = pixel / dimensions;

	Ray ray = make_ray(uv,
		pc.camera_position,
		pc.camera_forward,
		pc.camera_up,
		pc.camera_right,
		pc.properties.x,
		pc.properties.y
	);
	
	// Progressive rendering
	vec3 color = pathtracer(ray, MAX_DEPTH);
	vec3 pcolor = cast_color(frame.pixels[index]);
	pcolor = pow(pcolor, vec3(2.2));
	color = (color + pcolor * pc.present)/(pc.present + 1.0f);
	color = pow(color, vec3(1.0/2.2));
	frame.pixels[index] = cast_color(color);
}
