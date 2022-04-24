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
#define MAX_DEPTH 5

// Sample random point in triangle
float u = 0.0;
float v = 0.0;

vec3 sample_triangle(vec3 v1, vec3 v2, vec3 v3, float strata, float i)
{
	// Ignore random-ness if only 1 sample
	if (pc.samples_per_light == 1)
		return (v1 + v2 + v3) / 3.0;

	// Get random point in triangle
	vec2 uv = jitter2d(strata, i);
	if (uv.x + uv.y > 1.0)
		uv = vec2(1.0 - uv.x, 1.0 - uv.y);

	// Edge vectors
	vec3 e1 = v2 - v1;
	vec3 e2 = v3 - v1;

	vec3 p = v1 + uv.x * e1 + uv.y * e2;

	return p;
}

// Sample a light position from an area light
vec3 sample_light_position(uint li, int sample_i)
{
	uvec4 i = VERTEX_STRIDE * floatBitsToUint(lights.data[li + 1]);

	vec3 v1 = vertices.data[i.x].xyz;
	vec3 v2 = vertices.data[i.y].xyz;
	vec3 v3 = vertices.data[i.z].xyz;

	// Sample point from triangle
	return sample_triangle(
		v1, v2, v3,
		sqrt(pc.samples_per_light),
		sample_i
	);
}

// Reflect according to material and incoming ray
bool apply_bsdf(inout Ray r, Hit hit, inout float beta, inout float ior)
{
	// TODO: use int shading types for easy switching
	if (hit.mat.shading == SHADING_TYPE_DIFFUSE) {
		// Lambertian BSDF
		vec3 r_dir = random_hemi(hit.normal);
		r.direction = r_dir;
		r.origin = hit.point + hit.normal * 0.001;

		// Update beta
		beta *= dot(hit.normal, r_dir);
	} else if (hit.mat.shading == SHADING_TYPE_REFLECTION) {
		// (Perfect) Specular BSDF
		vec3 r_dir = reflect(r.direction, hit.normal);
		r.direction = r_dir;
		r.origin = hit.point + hit.normal * 0.001;

		// Update beta
		beta *= dot(hit.normal, r_dir);
	} else if (hit.mat.shading == SHADING_TYPE_REFRACTION) {
		// (Perfect) Transmissive BSDF
		vec3 r_dir = refract(r.direction, hit.normal, ior/hit.mat.ior);
		r.direction = r_dir;
		r.origin = hit.point - hit.normal * 0.001;

		// Update beta
		ior = hit.mat.ior;
	} else {
		// Invalid shading type
		return true;
	}

	return false;
}

// Color value at from a ray
vec3 color_at(Ray ray)
{
	// Light and camera paths
	vec3 light_contrs[MAX_DEPTH];
	vec3 camera_contr[MAX_DEPTH];

	vec3 light_vertices[MAX_DEPTH];
	vec3 camera_vertices[MAX_DEPTH];
	vec3 camera_normals[MAX_DEPTH];

	int light_objects[MAX_DEPTH];
	int camera_objects[MAX_DEPTH];

	// Initialize
	for (int i = 0; i < MAX_DEPTH; i++) {
		light_contrs[i] = vec3(0.0);
		camera_contr[i] = vec3(0.0);

		light_vertices[i] = vec3(0.0);
		camera_vertices[i] = vec3(0.0);
		camera_normals[i] = vec3(0.0);

		light_objects[i] = -1;
		camera_objects[i] = -1;
	}

	// Generate camera path
	Ray r = ray;

	float beta = 1.0;
	float ior = 1.0;

	int bounces = 0;
	for (; bounces < MAX_DEPTH; bounces++) {
		// Find closest object
		Hit hit = closest_object(r);

		// Value and position
		camera_contr[bounces] = beta * hit.mat.albedo;
		camera_vertices[bounces] = hit.point + hit.normal * 0.001;
		camera_normals[bounces] = hit.normal;
		camera_objects[bounces] = hit.object;

		// Special case intersection
		if (hit.object == -1 || hit.mat.shading == SHADING_TYPE_EMISSIVE) {
			camera_objects[bounces] = -1;
			break;
		}

		// Generating the new ray according to BSDF
		if (apply_bsdf(r, hit, beta, ior))
			break;

		// Russian roulette
		if (bounces > 2) {
			float q = max(1.0 - beta, 0.05);
			if (random() < q)
				break;
			beta /= (1.0 - q);
		}
	}

	// Total contribution
	vec3 total_contr = vec3(0.0);

	// Iterate over all lights
	for (int i = 0; i < pc.lights; i++) {
		// Iterate over requested # of shadow samples
		uint light_index = floatBitsToUint(light_indices.data[i]);

		vec3 light_contr = vec3(0.0);
		for (int j = 0; j < pc.samples_per_light; j++) {
			float beta = 1.0;
			float ior = 1.0;

			vec3 light_position = sample_light_position(light_index, j);

			// Reset light arrays
			for (int k = 0; k < MAX_DEPTH; k++) {
				light_contrs[k] = vec3(0.0);
				light_vertices[k] = vec3(0.0);
				light_objects[k] = -1;
			}

			// Always a first vertex
			light_contrs[0] = vec3(1.0);
			light_vertices[0] = light_position;
			light_objects[0] = -1;

			// Trace a light path
			vec3 dir = random_sphere();
			// Ray l = Ray(light_position + dir * 0.001, dir, 1.0, 1.0);
			r.origin = light_position + dir * 0.001;
			r.direction = dir;

			// Generate path
			int k = 1;
			for (; k < MAX_DEPTH; k++) {
				// Find closest object
				Hit hit = closest_object(r);

				// Value and position
				light_contrs[k] = beta * hit.mat.albedo;
				light_vertices[k] = hit.point + hit.normal * 0.001;
				light_objects[k] = hit.object;

				// Special case intersection
				if (hit.object == -1 || hit.mat.shading == SHADING_TYPE_EMISSIVE)
					break;

				// Generating the new ray according to BSDF
				if (apply_bsdf(r, hit, beta, ior))
					break;
			}

			// Calculate path contribution
			vec3 path_contr = vec3(0.0);

			for (int x = 0; x <= bounces; x++) {
				for (int y = 0; y <= k; y++) {
					// Special case
					if (camera_objects[x] == -1)  {
						path_contr += camera_contr[x];
						break;
					}

					vec3 ldir = normalize(light_vertices[y] - camera_vertices[x]);
					Ray visibility = Ray(
						camera_vertices[x],
						ldir,
						1.0, 1.0
					);

					Hit hit = closest_object(visibility);

					// TODO: use actual object id...
					if (hit.mat.shading == SHADING_TYPE_EMISSIVE
							|| hit.object == light_objects[y]) {
						float d = distance(light_vertices[y], camera_vertices[x]);
						float cos_theta = max(dot(ldir, camera_normals[x]), 0.0);
						path_contr += light_contrs[y]
							* camera_contr[x]
							* cos_theta
							* (5.0/d) * (2 * INV_PI * INV_PI);
						continue;
					}

					// TODO: special case for transmissive materials
					if (hit.object != -1
							&& hit.mat.shading == SHADING_TYPE_REFRACTION) {
						// Trace ray from light vertex
						//	to camera vertex (to hit the object)
						Ray light_to_camera = Ray(
							light_vertices[y],
							-ldir,
							1.0, 1.0
						);

						// TODO: biaS!!
						Hit light_hit_0 = closest_object(light_to_camera);

						if (light_hit_0.object == hit.object) {
							// Then connect to form caustics
							// after one more bounce
							// TODO: get accurate
							// iors
							float ior_ = 1.0;
							float beta_ = 1.0;

							apply_bsdf(visibility,
									hit,
									beta_,
									ior_);

							ior_ = 1.0;
							apply_bsdf(light_to_camera,
									light_hit_0,
									beta_,
									ior_);

							// Hit again (inside the object)
							Hit visibility_1 = closest_object(visibility);
							Hit light_hit_1 = closest_object(light_to_camera);

							// Distance is now a sum
							// of three distances
							float d1 = distance(light_vertices[y], light_hit_0.point);
							float d2 = distance(camera_vertices[x], hit.point);
							float d3 = distance(visibility_1.point, light_hit_1.point);

							// Connect each other
							// TODO: will need
							// additional cos_theta
							// factors
							// float d = d1 + d2 + d3;
							float d = distance(light_vertices[y], camera_vertices[x]);
							float cos_theta = max(dot(ldir, camera_normals[x]), 0.0);
							path_contr += light_contrs[y]
								* camera_contr[x]
								* cos_theta
								* (5.0/d) * (2 * INV_PI * INV_PI);
						}
					}
				}
			}

			// Update total light contribution
			light_contr += path_contr;
		}

		// Update total contribution
		total_contr += light_contr/float(pc.samples_per_light);
	}

	return clamp(total_contr, 0.0, 1.0);
}

// Gamma correction coefficients
const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = vec3(1.0/2.2);

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
		// Random offset
		vec2 offset = jitter2d(
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
		pcolor = pow(pcolor, gamma);
		color += pcolor * pc.present;
		color /= float(pc.present + pc.samples_per_pixel);
		color = pow(color, inv_gamma);
	} else {
		color /= float(pc.samples_per_pixel);
		color = pow(color, inv_gamma);
	}

	frame.pixels[index] = cast_color(color);
}
