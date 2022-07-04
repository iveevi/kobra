#version 450

// Import bindings
#include "bindings.h"

// Import all modules
#include "modules/ray.glsl"
#include "modules/bbox.glsl"
#include "modules/color.glsl"
#include "modules/layouts.glsl"
#include "modules/random.glsl"
#include "modules/material.glsl"
#include "modules/primitives.glsl"
#include "modules/environment.glsl"
#include "modules/intersect.glsl"
#include "modules/bvh.glsl"

// Maximum ray depth
#define MAX_DEPTH 10

// Sample random point in triangle
float u = 0.0;
float v = 0.0;

vec3 sample_triangle(vec3 v1, vec3 v2, vec3 v3, float strata, float i)
{
	// Ignore random-ness if only 1 sample
	if (pc.samples_per_light == 1)
		return (v1 + v2 + v3) / 3.0;

	// Get random vec in [0, 1]
	vec2 r = jitter2d(strata, i);
	float s = sqrt(r.x);

	float u = 1.0 - s;
	float v = r.y * s;

	// Edge vectors
	vec3 e1 = v2 - v1;
	vec3 e2 = v3 - v1;

	vec3 p = v1 + u * e1 + v * e2;

	return p;
}

// Center point of an area light
vec3 center_area_light(uint li)
{
	// Get light position
	uvec4 i = VERTEX_STRIDE * floatBitsToUint(lights.data[li + 1]);

	vec3 v1 = vertices.data[i.x].xyz;
	vec3 v2 = vertices.data[i.y].xyz;
	vec3 v3 = vertices.data[i.z].xyz;

	return (v1 + v2 + v3) / 3.0;
}

// Sample random point from an area light
vec3 sample_area_light(uint li, int si)
{
	// Get light position
	uvec4 i = VERTEX_STRIDE * floatBitsToUint(lights.data[li + 1]);

	vec3 v1 = vertices.data[i.x].xyz;
	vec3 v2 = vertices.data[i.y].xyz;
	vec3 v3 = vertices.data[i.z].xyz;

	// Get random point in triangle
	return sample_triangle(
		v1, v2, v3,
		sqrt(pc.samples_per_light), si
	);
}

// Power heuristic
float power_heuristic(float nf, float fpdf, float ng, float gpdf)
{
	float f = nf * fpdf;
	float g = ng * gpdf;

	return (f * f) / (f * f + g * g);
}

// Fresnel reflectance
float fresnel_dielectric(float cosi, float etai, float etat)
{
	// Swap if necessary
	cosi = clamp(cosi, -1.0, 1.0);
	if (cosi < 0.0) {
		cosi = -cosi;

		float tmp = etai;
		etai = etat;
		etat = tmp;
	}

	float sini = sqrt(max(0.0, 1.0 - cosi * cosi));

	float sint = etai / etat * sini;
	if (sint >= 1.0)
		return 1.0;

	float cost = sqrt(max(0.0, 1.0 - sint * sint));

	float r_parl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	float r_perp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));

	return (r_parl * r_parl + r_perp * r_perp) / 2.0;
}

// Invidual BSDF types
// TODO: header
#define BSDF(ftn) void bsdf_##ftn(in Hit hit,	\
		inout Ray ray,			\
		inout float pdf,		\
		inout vec3 beta,		\
		inout float ior)

BSDF(specular_reflection)
{
	float cosi = dot(ray.direction, hit.normal);
	float Fr = fresnel_dielectric(cosi, hit.mat.ior, ior);

	ray.direction = reflect(ray.direction, hit.normal);
	ray.origin = hit.point + hit.normal * 0.001;

	float cos_theta = abs(dot(ray.direction, hit.normal));

	pdf = 1;

	// TODO: fresnel conductors: if ior = 1, then
	// Fr is always equal to 0
	beta *= hit.mat.albedo/abs(cos_theta);
}

BSDF(specular_transmission)
{
	float cosi = dot(ray.direction, hit.normal);
	float Fr = fresnel_dielectric(cosi, hit.mat.ior, ior);

	float eta = ior/hit.mat.ior;
	ray.direction = refract(ray.direction, hit.normal, eta);
	ray.origin = hit.point - hit.normal * 0.001;

	float cos_theta = abs(dot(ray.direction, hit.normal));

	pdf = 1;
	ior = hit.mat.ior;
	beta *= (1 - Fr) * hit.mat.albedo / cos_theta;
}

BSDF(lambertian)
{
	// Assume diffuse
	ray.direction = cosine_weighted_hemisphere(hit.normal);
	ray.origin = hit.point + hit.normal * 0.001;

	pdf = INV_PI * dot(ray.direction, hit.normal);
	beta *= hit.mat.albedo * INV_PI;
}

// Sample a ray from BSDF
void sample_bsdf(in Hit hit, inout Ray ray,
		inout float pdf,
		inout vec3 beta,
		inout float ior)
{
	int shading = hit.mat.shading;
	if (is_type(shading, SHADING_REFLECTION | SHADING_TRANSMISSION)) {
		// Choose reflection or transmission randomly
		//	based on Fresnel reflectance
		float cosi = dot(ray.direction, hit.normal);
		float Fr = fresnel_dielectric(cosi, hit.mat.ior, ior);

		float rand = random();
		if (rand < Fr)
			bsdf_specular_reflection(hit, ray, pdf, beta, ior);
		else
			bsdf_specular_transmission(hit, ray, pdf, beta, ior);
	} else if (is_type(shading, SHADING_REFLECTION)) {
		bsdf_specular_reflection(hit, ray, pdf, beta, ior);
	} else if (is_type(shading, SHADING_TRANSMISSION)) {
		bsdf_specular_transmission(hit, ray, pdf, beta, ior);
	} else if (is_type(shading, SHADING_DIFFUSE)) {
		// TOOD: also microfacet diffuse is a sigma != 0
		bsdf_lambertian(hit, ray, pdf, beta, ior);
	} else {
		pdf = 0.0;
	}
	
	beta *= abs(dot(hit.normal, ray.direction)) / pdf;
}

// Get pdf of a direction from BSDF
float pdf_bsdf(in Hit hit, in Ray ray, vec3 wi)
{
	int shading = hit.mat.shading;
	if (is_type(shading, SHADING_DIFFUSE)) {
		// Assume diffuse
		// TODO: check same hemisphere
		return INV_PI * dot(wi, hit.normal);
	}
	
	return 0.0;
}

// Total color for a ray
vec3 color_at(Ray ray)
{
	// Camera path
	Ray	rays[MAX_DEPTH];
	Hit	hits[MAX_DEPTH];
	vec3	betas[MAX_DEPTH];
	int	depth = 1;

	// Generate camera path
	vec3 beta = vec3(1.0);

	// Index of refraction
	float ior = 1.0;

	Ray r = ray;
	for (int i = 0; i < MAX_DEPTH; i++, depth++) {
		// Find closest object, and store hit stuff
		hits[i] = trace(r);
		rays[i] = r;
		betas[i] = beta;

		// Special case intersection
		if (hits[i].object == -1)
			break;

		if (hits[i].mat.shading == SHADING_EMISSIVE) {
			// Hard code the albedo for now,
			// later retrieve properties from
			// the light object index, etc
			hits[i].mat.albedo = vec3(1.0);
			break;
		}
		
		// Sample BSDF
		float pdf = 0.0;
		sample_bsdf(hits[i], r, pdf, beta, ior);

		if (pdf == 0.0)
			break;

		// Russian roulette
		if (i > 2) {
			float q = max(1.0 - beta.y, 0.05);
			if (random() < q)
				break;
			beta /= (1.0 - q);
		}
	}

	// Total contribution
	vec3 contribution = vec3(0.0);

	// Lights
	float bsdf_samples = pc.samples_per_surface;
	float light_samples = pc.samples_per_light;

	for (int n = 0; n < pc.lights; n++) {
		// TODO: account for samples per light (loop here)
		uint light_index = floatBitsToUint(light_indices.data[n]);
		int light_object = int(floatBitsToUint(lights.data[light_index + 1].w));

		// Get light position
		vec3 light_position = sample_area_light(light_index, 0);

		// TODO: account for BSDF sampling

		// Light path data
		Ray	light_rays[MAX_DEPTH];
		Hit	light_hits[MAX_DEPTH];
		vec3	light_betas[MAX_DEPTH];
		int	light_depth = 1;

		// Generate light path
		vec3 light_beta = vec3(1.0);

		// Index of refraction
		float light_ior = 1.0;

		// Staring ray
		// TODO: maybe choose an initial direction that is cosine
		// weighted towards the first hit?
		Ray lr = Ray(light_position, random_sphere(), 0.0, 0.0);

		for (int i = 0; i < 1; i++, light_depth++) {
			// Find closest object, and store hit stuff
			light_hits[i] = trace(lr);
			light_rays[i] = lr;
			light_betas[i] = light_beta;

			// Sample BSDF
			float pdf = 0.0;
			sample_bsdf(light_hits[i], lr, pdf, light_beta, light_ior);

			if (pdf == 0.0)
				break;

			// Russian roulette
			if (i > 2) {
				float q = max(1.0 - light_beta.y, 0.05);
				if (random() < q)
					break;
				light_beta /= (1.0 - q);
			}
		}

		// Now compute the contributions
		int count = 0;
		for (int i = 0; i < depth; i++) {
			// Get hit
			Hit hit = hits[i];
			Ray ray = rays[i];

			if (hit.mat.shading == SHADING_EMISSIVE) {
				contribution += hit.mat.albedo;

				// Should be the last one anyway
				break;
			}

			vec3 cam_hit = hits[i].point + hits[i].normal * 0.001;
			vec3 beta = betas[i];
			vec3 albedo = hits[i].mat.albedo;

			for (int j = 0; j < light_depth; j++) {
				// Get light hit
				vec3 light_hit = light_rays[j].origin;
				
				// Temp variables
				float d = distance(light_hit, cam_hit);
				vec3 lcolor = light_betas[j];

				// BSDF sampling
				float pdf = 0.0;
				float ior = 1.0;

				Ray r = ray;
				sample_bsdf(hit, r, pdf, beta, ior);

				// Trace and see if it hits close to the light
				Hit bsdf_hit = trace(r);

				float dist = distance(bsdf_hit.point, light_hit);
				if (dist <= 0.0025) {
					float cos_theta = max(0.0, dot(hits[i].normal, r.direction));

					float mis_weight = power_heuristic(
						bsdf_samples, pdf,
						light_samples, 0.25 * INV_PI
					);

					contribution += lcolor * albedo
						* cos_theta * mis_weight
						* (1/(d * d)) * (1/pdf);
					count++;
				}

				// Visibility ray (light sampling)
				vec3 vis_dir = normalize(light_hit - cam_hit);
				Ray vis_ray = Ray(cam_hit, vis_dir, 0.0, 0.0);
			
				pdf = pdf_bsdf(hit, ray, vis_dir);
				if (pdf == 0.0)
					continue;

				// Check if visible
				Hit vis_hit = trace(vis_ray);

				// Check that the hit points are close
				dist = distance(vis_hit.point, light_hit);
				if (dist <= 0.0025) {
					float cos_theta = max(0.0, dot(hits[i].normal, vis_dir));

					// Compute the contribution
					float mis_weight = power_heuristic(
						light_samples,
						0.25 * INV_PI,
						bsdf_samples, pdf
					);

					contribution += lcolor * beta * albedo
						* cos_theta * (1/(d * d))
						* mis_weight * (4 * PI);
					count++;
				}
			}
		}
	}

	// TODO: clamp at the film (final), not at the camera
	return clamp(contribution, 0.0, 1.0);
}

// Mitchell-Netravali 1D
float mitchell_netravali(float x, float b, float c)
{
	x = abs(2 * x);
	if (x > 1) {
		return (1.0/6.0) * (x * x * x * (-b - 6 * c)
			+ x * x * (6 * b + 30 * c)
			+ x * (-12 * b - 48 * c)
			+ 8 * b + 24 * c);
	}

	return (1.0/6.0) * (x * x * x * (12 - 9 * b - 6 * c)
		+ x * x * (-18 + 12 * b + 6 * c)
		+ 6 - 2 * b);
}

// Image reconstruction
vec3 filter_out(vec3 color, vec2 offset)
{
	/* Mitchell filter
	float a = 2.0 / 3.0;
	float b = 1.0 / 3.0;
	return mitchell_netravali(offset.x, a, b)
		* mitchell_netravali(offset.y, a, b)
		* color; */

	return color;
}

/* Halton sequence
float halton(int index, int base)
{
	float f = 1.0;
	float r = 0.0;
	while (index > 0) {
		f = f / base;
		r = r + f * (index % base);
		index = floor(index / base);
	}
	return r;
}

vec2 roffset(int i)
{
	return vec2(halton(i, 2), halton(i, 3));
} */

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y + pc.xoffset;
	uint x0 = gl_WorkGroupID.x + pc.yoffset;

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

	// Accumulate color
	vec3 color = vec3(0.0);

	vec2 dimensions = vec2(pc.width, pc.height);
	for (int i = 0; i < pc.samples_per_pixel; i++) {
		// Create the ray
		vec2 offset = random_sphere().xy/2.0f;
		vec2 pixel = vec2(x0, y0) + offset;
		vec2 uv = pixel / dimensions;

		Ray ray = make_ray(uv,
			pc.camera_position,
			pc.camera_forward,
			pc.camera_up,
			pc.camera_right,
			pc.properties.x,
			pc.properties.y
		);

		// Light transport
		color += filter_out(color_at(ray), offset);
	}

	if (pc.accumulate > 0) {
		vec3 pcolor = cast_color(frame.pixels[index]);
		pcolor = pow(pcolor, vec3(2.2));
		color += pcolor * pc.present;
		color /= float(pc.present + pc.samples_per_pixel);
		color = pow(color, vec3(1/2.2));
	} else {
		color /= float(pc.samples_per_pixel);
		color = pow(color, vec3(1/2.2));
	}

	frame.pixels[index] = cast_color(color);
}
