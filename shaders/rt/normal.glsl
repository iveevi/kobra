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

vec3 color_at(Ray ray)
{
	Hit hit = closest_object(ray);
	if (hit.object == -1)
		return vec3(0.0);
	return hit.normal * 0.5 + 0.5;
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
	float rx = fract(sin(x0 * 1234.56789 + y0) * PHI);
	float ry = fract(sin(y0 * 9876.54321 + x0));

	// Accumulate color
	// TODO: skip this for normals...
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
		color += pcolor * pc.present;
		color /= float(pc.present + pc.samples_per_pixel);
	} else {
		color /= float(pc.samples_per_pixel);
	}

	frame.pixels[index] = cast_color(color);
}
