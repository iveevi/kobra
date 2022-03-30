#version 450

// Import bindings
#include "mesh_bindings.h"

layout (set = 0, binding = MESH_BINDING_PIXELS, std430) buffer Pixels
{
	uint pixels[];
} frame;

// TODO: should be a push constant
layout (set = 0, binding = MESH_BINDING_VIEWPORT, std430) buffer Viewport
{
	uint width;
	uint height;
} viewport;

layout (set = 0, binding = MESH_BINDING_VERTICES, std430) buffer Vertices
{
	vec4 data[];
} vertices;

layout (set = 0, binding = MESH_BINDING_TRIANGLES, std430) buffer Triangles
{
	vec4 data[];
} triangles;

// Push constants
layout (push_constant) uniform PushConstants
{
	uint	triangles;

	// Camera
	vec3 camera_position;
	vec3 camera_forward;
	vec3 camera_up;
	vec3 camera_right;
	
	// scale, aspect
	vec4 properties;
} pc;

// Import other headers
#include "common/color.glsl"
#include "common/ray.glsl"
#include "common/intersect.glsl"

// Intersection
Intersection ray_intersect(Ray ray, uint index)
{
	float ia = triangles.data[index].x;
	float ib = triangles.data[index].y;
	float ic = triangles.data[index].z;

	uint a = floatBitsToUint(ia);
	uint b = floatBitsToUint(ib);
	uint c = floatBitsToUint(ic);

	vec3 v1 = vertices.data[a].xyz;
	vec3 v2 = vertices.data[b].xyz;
	vec3 v3 = vertices.data[c].xyz;
	
	Triangle triangle = Triangle(v1, v2, v3);

	// Intersect ray with sphere
	return intersect_shape(ray, triangle);
}

// Get closest object
float closest_object(Ray ray)
{
	int min_index = -1;

	// Starting time
	Intersection mini = Intersection(
		1.0/0.0, vec3(0.0),
		mat_default()
	);

	// If the ray is null
	if (ray.direction == vec3(0.0))
		return -1.0;

	for (int i = 0; i < pc.triangles; i++) {
		// Get object
		Intersection it = ray_intersect(ray, i);

		// If intersection is valid, update minimum
		if (it.time > 0.0 && it.time < mini.time) {
			min_index = i;
			mini = it;
		}
	}

	if (min_index == -1)
		return -1.0;

	return mini.time;
}

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	// Return if out of bounds
	if (y0 >= viewport.height || x0 >= viewport.width)
		return;

	// Create the ray
	vec2 pixel = vec2(x0 + 0.5, y0 + 0.5);
	vec2 dimensions = vec2(viewport.width, viewport.height);
	vec2 uv = pixel / dimensions;

	Ray ray = make_ray(uv,
		pc.camera_position,
		pc.camera_forward,
		pc.camera_up,
		pc.camera_right,
		pc.properties.x,
		pc.properties.y
	);

	float t = closest_object(ray);

	// Light transport
	// vec3 color = vertices.data[0].xyz + triangles.data[pc.triangles - 1].xyz;
	vec3 color = vec3(0.0);
	if (t > 0.0)
		color = vec3(1.0);
	
	// Get index
	uint index = y0 * viewport.width + x0;
	frame.pixels[index] = cast_color(color);
}
