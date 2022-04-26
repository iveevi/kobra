#version 450

// Import bindings
#include "bindings.h"

// Import all modules
#include "../../include/types.hpp"
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

// Sample a direction from BSDF
void sample_bsdf(in Hit hit, inout Ray ray, inout float pdf)
{
	float shading = hit.mat.shading;
	if (shading == SHADING_REFLECTION) {
		ray.direction = reflect(ray.direction, hit.normal);
		pdf = 1.0;
	} else {
		// Assume diffuse
		ray.direction = cosine_weighted_hemisphere(hit.normal);
		pdf = INV_PI * dot(ray.direction, hit.normal);
	}
}

// Get pdf of a direction from BSDF
float pdf_bsdf(in Hit hit, in Ray ray, vec3 wi)
{
	float shading = hit.mat.shading;
	if (shading == SHADING_REFLECTION) {
		vec3 refl = reflect(ray.direction, hit.normal);

		if (length(refl - wi) < 0.001)
			return 1.0;
		else
			return 0.0;
	}

	// Assume diffuse
	return 0.5 * INV_PI;
}

// Direct illumination
vec3 direct_illumination(Hit hit, Ray ray)
{
	// Direct light contribution
	vec3 direct_contr = vec3(0.0);

	// Direct illumination
	for (int i = 0; i < pc.lights; i++) {
		uint light_index = floatBitsToUint(light_indices.data[i]);

		// Ray to use
		Ray r = ray;

		//////////////////////////////////
		// Multiple importance sampling //
		//////////////////////////////////

		// Sampled on BSDF
		float bsdf_samples = pc.samples_per_surface;
		// if (hit.mat.shading == SHADING_REFLECTION)
		//	bsdf_samples = 1.0;

		float pdf = 0.0;

		// Itetrate over samples
		for (int j = 0; j < bsdf_samples; j++) {
			// Sample BSDF
			sample_bsdf(hit, r, pdf);

			// Get intersect
			Hit shadow_hit = closest_object(r);

			// Assume success if emissive
			vec3 bsdf_contr = vec3(1.0);
			if (shadow_hit.mat.shading != SHADING_EMISSIVE)
				continue;

			float d = distance(shadow_hit.point, hit.point);
			float cos_theta = dot(hit.normal, r.direction);
			direct_contr += bsdf_contr * hit.mat.albedo
				* cos_theta * (5/d)
				* power_heuristic(bsdf_samples, pdf, float(pc.samples_per_light), 0.25 * INV_PI)
				* (1/pdf);
			
			// TODO: caustics from here
		}

		float inv_lsamples = 1.0/pc.samples_per_light;
		for (int j = 0; j < pc.samples_per_light; j++) {
			vec3 light_position = sample_area_light(light_index, j);

			// Try to connect to the light
			vec3 pos = hit.point + hit.normal * 0.001;
			vec3 dir = normalize(light_position - pos);

			float pdf = pdf_bsdf(hit, r, dir);
			if (pdf == 0.0)
				continue;

			// TODO: remove the extra arguments
			Ray shadow_ray = Ray(pos, dir, 1.0, 1.0);

			Hit shadow_hit = closest_object(shadow_ray);
			if (shadow_hit.mat.shading == SHADING_EMISSIVE) {
				// Light contribution directly
				float d = distance(light_position, hit.point);
				float cos_theta = max(0.0, dot(hit.normal, dir));
				vec3 lcolor = vec3(1.0);

				// TODO: get the pdf from anotehr function
				direct_contr += lcolor * hit.mat.albedo
					* cos_theta * (5/d)
					* power_heuristic(float(pc.samples_per_light), 0.25 * INV_PI, bsdf_samples, pdf)
					* inv_lsamples * (4 * PI);
			}

			/* else if (shadow_hit.mat.shading == SHADING_TRANSMISSION) {
				// Light contribution from refractive caustics

				// Make sure that the light to vertex ray hits
				//	the same object as that the shadow ray
				vec3 dir = -shadow_ray.direction;
				shadow_ray.origin = light_position + dir * 0.001;
				shadow_ray.direction = dir;

				Hit light_hit = closest_object(shadow_ray);
				if (light_hit.object != shadow_hit.object)
					continue;

				// Apply lighting
				float d1 = distance(hit.point, shadow_hit.point);
				float d2 = distance(shadow_hit.point, light_hit.point);
				float d3 = distance(light_hit.point, light_position);

				float d = d1 + d2 + d3;
				float cos_theta = max(0.0, dot(hit.normal, dir));
				vec3 lcolor = vec3(1.0);

				direct_contr += lcolor * hit.mat.albedo
					* cos_theta * (5/d)
					* power_heuristic(float(pc.samples_per_light), 0.25 * INV_PI, bsdf_samples, pdf)
					* inv_lsamples * (4 * PI);
			} */
		}
	}

	// Plus environment
	// TODO: enable/disable feature
	if (true) {
		// Ray to use
		Ray r = ray;

		//////////////////////////////////
		// Multiple importance sampling //
		//////////////////////////////////
		
		// Sampled on BSDF
		float bsdf_samples = pc.samples_per_surface;
		// if (hit.mat.shading == SHADING_REFLECTION)
		//	bsdf_samples = 1.0;

		float pdf = 0.0;

		// Itetrate over samples
		for (int j = 0; j < bsdf_samples; j++) {
			// Sample BSDF
			sample_bsdf(hit, r, pdf);

			// Get intersect
			Hit shadow_hit = closest_object(r);

			// Assume success if emissive
			if (shadow_hit.object != -1)
				continue;
			
			vec3 bsdf_contr = sample_environment(r);
			float d = distance(shadow_hit.point, hit.point);
			float cos_theta = dot(hit.normal, r.direction);
			direct_contr += bsdf_contr * hit.mat.albedo
				* cos_theta * (5/d)
				* power_heuristic(bsdf_samples, pdf, float(pc.samples_per_light), 0.25 * INV_PI)
				* (1/pdf);
			
			// TODO: caustics from here
		}

		float samples = pc.samples_per_light;
		float inv_lsamples = 1.0/samples;
		for (int j = 0; j < samples; j++) {
			// Fixed distance away
			const float d = 100;

			// Try to connect to the light
			vec3 pos = hit.point + hit.normal * 0.001;
			vec3 dir = random_sphere();

			float pdf = pdf_bsdf(hit, r, dir);
			if (pdf == 0.0)
				continue;

			// TODO: remove the extra arguments
			Ray shadow_ray = Ray(pos, dir, 1.0, 1.0);

			Hit shadow_hit = closest_object(shadow_ray);
			if (shadow_hit.object != -1) {
				// Light contribution directly
				float cos_theta = max(0.0, dot(hit.normal, dir));
				vec3 lcolor = vec3(1.0);

				// TODO: get the pdf from anotehr function
				direct_contr += lcolor * hit.mat.albedo
					* cos_theta * (5/d)
					* power_heuristic(samples, 0.25 * INV_PI, bsdf_samples, pdf)
					* inv_lsamples * (4 * PI);
			}

			// TODO: caustics later
		}
	}

	return direct_contr;
}

// Color value at from a ray
int bounce_count = 0;

vec3 color_at(Ray ray)
{
	vec3 contribution = vec3(0.0);
	float beta = 1.0;
	float ior = 1.0;

	Ray r = ray;
	
	float count = 0;
	for (int i = 0; i < MAX_DEPTH; i++) {
		count++;

		// Find closest object
		Hit hit = closest_object(r);

		// Special case intersection
		if (hit.object == -1 || hit.mat.shading == SHADING_EMISSIVE) {
			contribution += hit.mat.albedo;
			break;
		}

		// Add bounce count
		bounce_count++;

		// Direct illumination
		vec3 direct_contr = direct_illumination(hit, r)/PI;
		contribution += beta * direct_contr;

		// Generating the new ray according to BSDF
		if (hit.mat.shading == SHADING_DIFFUSE) {
			// Lambertian BSDF
			vec3 r_dir = cosine_weighted_hemisphere(hit.normal);
			r = Ray(
				hit.point + hit.normal * 0.001,
				r_dir, 1.0, 1.0
			);

			// Update beta
			beta *= dot(hit.normal, r_dir);
		} else if (hit.mat.shading == SHADING_REFLECTION) {
			// (Perfect) Specular BSDF
			vec3 r_dir = reflect(r.direction, hit.normal);
			r.direction = r_dir;
			r.origin = hit.point + hit.normal * 0.001;

			// Update beta
			beta *= dot(hit.normal, r_dir);
		} else if (hit.mat.shading == SHADING_TRANSMISSION) {
			// (Perfect) Transmissive BSDF
			vec3 r_dir = refract(r.direction, hit.normal, ior/hit.mat.ior);
			r.direction = r_dir;
			r.origin = hit.point - hit.normal * 0.001;

			// Update beta
			ior = hit.mat.ior;
		} else {
			// Assume no interaction
			break;
		}

		// Russian roulette
		if (i > 2) {
			float q = max(1.0 - beta, 0.05);
			if (random() < q)
				break;
			beta /= (1.0 - q);
		}
	}

	return clamp(contribution, 0.0, 1.0);
}

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

		// Light transport
		color += color_at(ray);
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

	///////////////////////////////
	// Denoising post processing //
	///////////////////////////////

	/* Variation in neighboring pixels
	vec3 variation = vec3(0.0);
	float variance = 0.0;

	int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
	int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

	for (int i = 0; i < 8; i++) {
		int x = int(x0) + dx[i];
		int y = int(y0) + dy[i];

		if (x >= 0 && x < pc.width && y >= 0 && y < pc.height) {
			uint index = y * pc.width + x;
			vec3 pc = cast_color(frame.pixels[index]);
			variation += (pc - color);
			variance += length(pc - color);
		}
	}

	variation /= float(8);
	variance = (variance - length(variation)) / float(8);

	// If variance is too high, leave it
	// If variance is low, leave  it
	// Sweet spot to add variance is in the middle
	float s = smoothstep(0.0, 1.0, variance);
	if (s > 0.1 && s < 0.4 && bounce_count > 0)
		color += variation; */

	frame.pixels[index] = cast_color(color);
}
