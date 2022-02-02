#version 430

#include "ray.glsl"
#include "shapes.glsl"
#include "intersect.glsl"

// TODO: replace version with command line argument
// TODO: header system

layout (set = 0, binding = 0, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = 1, std430) buffer World
{
	uint objects;
	uint lights;
	uint background;

	uint width;
	uint height;

	vec3 camera;

	// TODO: make a camera structure
	// plus transform
	
	// fov, scale, aspect
	float fov;
	float scale;
	float aspect;

	vec4 data[16];
} world;

layout (set = 0, binding = 2, std430) buffer Objects
{
	vec4 data[];
} objects;

// Get index of cloests object
int closest_object(Ray ray)
{
	int min_index = -1;
	float min_distance = 1.0/0.0;

	for (int i = 0; i < world.objects; i++) {
		vec3 p = objects.data[0].xyz;
		float r = objects.data[0].w;

		// p = vec3(0, 0, 0);
		// r = 6;

		Sphere sphere = Sphere(p, r);

		float d = intersect(sphere, ray);
		if (d > 0 && d < min_distance) {
			min_distance = d;
			min_index = i;
		}
	}

	return min_index;
}

void main()
{
	// Calculate initial x and y and offset indexes
	uint sphere_color = 0x00FF00;

	uint color;

	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	uint ysize = gl_NumWorkGroups.y;
	uint xsize = gl_NumWorkGroups.x;

	for (uint y = y0; y < world.height; y += ysize) {
		for (uint x = x0; x < world.width; x += xsize) {
			uint index = y * world.width + x;

			vec2 uv = vec2(x + 0.5, y + 0.5) / vec2(world.width, world.height);

			Ray ray = make_ray(uv, world.camera, world.scale, world.aspect);
			
			int object_index = closest_object(ray);
			if (object_index >= 0) {
				color = sphere_color;
			} else {
				color = world.background;
			}

			frame.pixels[index] = color;
		}
	}
}
