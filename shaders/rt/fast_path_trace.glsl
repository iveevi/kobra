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
#include "modules/constants.glsl"
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

	// Get random point in triangle
	vec2 uv = jitter2d(vec2(i), strata, i);
	if (uv.x + uv.y > 1.0)
		uv = vec2(1.0 - uv.x, 1.0 - uv.y);

	// Edge vectors
	vec3 e1 = v2 - v1;
	vec3 e2 = v3 - v1;

	vec3 p = v1 + uv.x * e1 + uv.y * e2;

	// Update seed
	ray_seed = fract((ray_seed + 1.0) * PHI);

	return p;
}

vec2 point_light_contr(Hit hit, vec3 pos)
{
	// Light intensity
	float d = distance(pos, hit.point);
	float intensity = 5.0/d;

	// Lambertian
	vec3 light_direction = normalize(pos - hit.point);
	float diffuse = max(dot(light_direction, hit.normal), 0.0);

	return vec2(intensity * diffuse, d);
}

vec3 point_light_full(Hit hit, Ray ray, vec3 pos)
{
	// Sum of colors
	vec3 color = vec3(0.0);

	// Create shadow ray
	Ray shadow_ray = Ray(
		hit.point + hit.normal * 0.001,
		normalize(pos - hit.point),
		1.0, 1.0
	);

	// Array of rays
	Ray rays[MAX_DEPTH];
	rays[0] = shadow_ray;

	int index = 0;
	int depth = 1;

	// Light contribution
	vec2 light_contr = point_light_contr(hit, pos);

	// Oldest hit
	Hit old_hit = hit;

	// Loop until no more light
	while (index < MAX_DEPTH && index < depth) {
		// Get ray and hit
		Ray r = rays[index];
		Hit h = closest_object(r);

		if (h.object == -1) {
			// No hit, full contribution
			color += r.contr * light_contr.x * hit.mat.albedo;
		} else if (h.mat.shading == SHADING_TYPE_EMISSIVE
				|| h.time > light_contr.y) {
			// Hit object counts as light
			color += r.contr * light_contr.x * hit.mat.albedo;
		} /* else if (h.mat.shading == SHADING_TYPE_REFLECTION) {
			// TODO: is reflectivity even needed here?
			// Reflection doesnt preserve much light
			float contr = 0.85;

			// Possibly a little bit of light from reflection
			Ray nray = Ray(
				h.point + h.normal * 0.001,
				reflect(r.direction, h.normal),
				1.0, contr * r.contr
			);

			old_hit = h;
			rays[depth] = nray;
			depth++;
		} */ else if (h.mat.shading == SHADING_TYPE_REFRACTION) {
			// Refraction preserves more light
			float contr = 0.9;

			// Possibly a little bit of light from refraction
			// TODO: preserve color of the object (tint)
			// TODO: use the correct ior?
			Ray nray = Ray(
				h.point - h.normal * 0.001,
				refract(r.direction, h.normal, 1.0),
				h.mat.ior, contr * r.contr
			);

			old_hit = h;
			rays[depth] = nray;
			depth++;

			// return vec3(h.mat.ior - 1);
		}

		index++;
	}

	return color;
}

vec3 single_area_light_contr(Hit hit, Ray ray, vec3 v1, vec3 v2, vec3 v3)
{
	vec3 color = vec3(0.0);
	for (int i = 0; i < pc.samples_per_light; i++) {
		// Sample point from triangle
		vec3 light_position = sample_triangle(
			v1, v2, v3,
			sqrt(pc.samples_per_light), i
		);

		color += point_light_full(hit, ray, light_position);
	}

	return color/float(pc.samples_per_light);
}

vec3 single_area_light_contr(Hit hit, Ray ray, int i)
{
	// Unique (but same) seed for each light
	ray_seed = fract(sin(i * 4325874.3423) * 4398.4324);
	// ray_seed = 0.0;

	// TODO: samlpe method for each light
	uint light_index = floatBitsToUint(light_indices.data[i]);

	float a = lights.data[light_index + 1].x;
	float b = lights.data[light_index + 1].y;
	float c = lights.data[light_index + 1].z;

	uint ia = floatBitsToUint(a);
	uint ib = floatBitsToUint(b);
	uint ic = floatBitsToUint(c);

	vec3 v1 = vertices.data[2 * ia].xyz;
	vec3 v2 = vertices.data[2 * ib].xyz;
	vec3 v3 = vertices.data[2 * ic].xyz;

	return single_area_light_contr(hit, ray, v1, v2, v3);
}

vec3 light_contr(Hit hit, Ray ray)
{
	vec3 contr = vec3(0.0);

	// Iterate over all lights
	for (int i = 0; i < pc.lights; i++)
		contr += single_area_light_contr(hit, ray, i);

	///////////////////////////////////
	// Sample light from environment //
	///////////////////////////////////

	// First, get the reflected ray
	vec3 r = reflect(ray.direction, hit.normal);
	Ray refl = Ray(hit.point + hit.normal * 0.001, r, 1.0, 1.0);

	// If the environment is not occlude, add it
	Hit env_hit = closest_object(refl);
	if (env_hit.object != -1)
		return contr/float(pc.lights);

	contr += sample_environment_blur(refl);
	return contr/float(pc.lights + 1);
}

vec3 color_at(Ray ray)
{
	/* TODO: consider adding a normal map mode
	Hit hit = closest_object(ray);
	if (hit.object == -1)
		return vec3(0.0);
	return hit.normal * 0.5 + 0.5; */

	// Array of rays
	Ray rays[MAX_DEPTH];
	rays[0] = ray;

	int index = 0;
	int depth = 1;

	// Total color
	vec3 color = vec3(0.0);

	// Iterate over all rays
	while (index < MAX_DEPTH && index < depth) {
		Ray r = rays[index];
		Hit hit = closest_object(r);

		if (hit.object == -1) {
			// No hit
			color += hit.mat.albedo;
		} else if (hit.mat.shading == SHADING_TYPE_EMISSIVE) {
			// Emissive
			color += hit.mat.albedo;
		} else {
			// Hit
			color += r.contr * light_contr(hit, r);

			// Reflection
			if (hit.mat.shading == SHADING_TYPE_REFLECTION
					&& depth < MAX_DEPTH) {
				// Reflection ray
				vec3 r_dir = reflect(r.direction, hit.normal);
				Ray r_ray = Ray(
					hit.point + hit.normal * 0.001,
					r_dir, 1.0, 0.8 * r.contr
				);

				rays[depth] = r_ray;
				depth++;
			}

			// Refraction
			if (hit.mat.shading == SHADING_TYPE_REFRACTION
					&& depth < MAX_DEPTH) {
				// Refraction ray
				vec3 r_dir = refract(
					r.direction,
					hit.normal,
					r.ior/hit.mat.ior
				);

				Ray r_ray = Ray(
					hit.point - hit.normal * 0.001,
					r_dir, 1.0, 0.8 * r.contr
				);

				rays[depth] = r_ray;
				depth++;
			}
		}

		index++;
	}

	// return vec3(sin(0.1 * index));
	return clamp(color, 0.0, 1.0);
}

// TODO: color wheel (if needed) goes in a header

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
	float rx = fract(sin(x0 * 1234.56789 + y0) * PHI);
	float ry = fract(sin(y0 * 9876.54321 + x0));
	ray_seed = rx + ry;

	// Accumulate color
	vec3 color = vec3(0.0);

	vec2 dimensions = vec2(pc.width, pc.height);
	for (int i = 0; i < pc.samples_per_pixel; i++) {
		// Random offset
		vec2 offset = jitter2d(
			vec2(x0, y0),
			sqrt(pc.total),
			i + pc.present
		);

		// Create the ray
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

	frame.pixels[index] = cast_color(color);
}
