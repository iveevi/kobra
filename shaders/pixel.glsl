#version 430

#include "ray.glsl"

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

	float data[];
} world;

// Sphere intersection with ray
struct Sphere {
	vec3 center;
	float radius;
};

float intersect(Sphere s, Ray r)
{
	vec3 oc = r.origin - s.center;
	float a = dot(r.direction, r.direction);
	float b = 2.0 * dot(oc, r.direction);
	float c = dot(oc, oc) - s.radius * s.radius;
	float d = b * b - 4.0 * a * c;

	if (d < 0.0)
		return -1.0;
	
	float t1 = (-b - sqrt(d)) / (2.0 * a);
	float t2 = (-b + sqrt(d)) / (2.0 * a);

	return min(t1, t2);
}

// NOTE: pixel format is BGRA, not RGBA
void main()
{
	uint sphere_color = 0x00FF00;

	uint color;
	for (int y = 0; y < world.height; y++) {
		for (int x = 0; x < world.width; x++) {
			uint index = y * world.width + x;

			vec2 uv = vec2(x + 0.5, y + 0.5) / vec2(world.width, world.height);

			Ray ray = make_ray(uv, world.camera, world.scale, world.aspect);

			float t = intersect(Sphere(vec3(0.0, 0.0, 0.0), 6.0), ray);

			// color = world.background;
			if (t >= 0.0)
				color = sphere_color;
			else
				color = world.background;

			frame.pixels[index] = color;
		}
	}
}
