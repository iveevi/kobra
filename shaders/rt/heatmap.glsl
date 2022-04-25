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

	// Vector of dimensions
	vec2 dimensions = vec2(pc.width, pc.height);

	// Create the ray
	vec2 uv = vec2(x0, y0) / dimensions;

	Ray ray = make_ray(uv,
		pc.camera_position,
		pc.camera_forward,
		pc.camera_up,
		pc.camera_right,
		pc.properties.x,
		pc.properties.y
	);

	// Iterate through BVHs boxes and primitives
	int node = 0;
	int count = 0;
	int hits = 0;

	while (node != -1) {
		count++;
		if (leaf(node)) {
			// Get object index
			int index = object(node);

			// Get object
			Intersection it = ray_intersect(ray, index);
			if (it.time > 0.0)
				hits++;
			
			// Only considering non-leaf nodes
			node = miss(node);
		} else {
			// Get bounding box
			BoundingBox box = bbox(node);

			// Check if ray intersects bounding box
			float t = intersect_box(ray, box);
			bool inside = in_box(ray.origin, box);

			if (t >= 0.0 || inside) {
				// Intersected
				node = hit(node);
			} else {
				// Missed
				node = miss(node);
			}
		}
	}

	// Heatmap color
	float s = PI * smoothstep(0.0, 1.0, count/float(1000))/2;
	vec3 color = vec3(sin(s), sin(2 * s), cos(s));

	if (hits > 0)
		color = mix(color, vec3(0, 1, 1), 0.2);

	frame.pixels[index] = cast_color(color);
}
