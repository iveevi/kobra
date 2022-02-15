#version 430

// TODO: rename file

#include "../include/types.h"

#include "ray.glsl"
#include "intersect.glsl"
#include "color.glsl"

// TODO: replace version with command line argument

layout (set = 0, binding = 0, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = 1, std430) buffer World
{
	uint objects;
	uint lights;
	// uint background;

	uint width;
	uint height;

	vec4 camera;
	vec4 cforward;
	vec4 cup;
	vec4 cright;

	// TODO: make a camera structure
	// plus transform
	
	// fov, scale, aspect
	/* float fov;
	float scale;
	float aspect; */
	vec4 tunings;

	// TODO: indices for objects, lights, background
	uint indices[];
} world;

layout (set = 0, binding = 2, std430) buffer Objects
{
	// Object layout:
	// vec4 descriptor
	// 	[x] Shape: sphere, plane, mesh, etc.
	//	[y] Shading: blinn-phong, normal, flat, etc.
	//	[z, w] Size and offset of object data
	// vec4 color (TODO: ?)
	vec4 data[];
} objects;

layout (set = 0, binding = 3, std430) buffer Lights
{
	// Light layout:
	// vec4 descriptor (point, directional, etc. + size and offset)
	// vec4 color
	vec4 data[];
} lights;

// Array of materials
layout (set = 0, binding = 4, std430) buffer Materials
{
	// Material layout:
	// vec4 descriptor (color, shading)
	vec4 data[];
} materials;

// Closest object information
// TODO: this should be obslete
struct Hit {
	int	object;
	float	time;
	vec3	point;
	vec3	normal;
	vec3	color;
	float	shading;	// Shading type
};

// Background color corresponding to ray
// corresponds to a sky like gradient
vec3 background(Ray ray)
{
	vec3 direction = normalize(ray.direction);
	float t = 0.5 * (1.0 + direction.y);
	return  mix(
		vec3(1.0, 1.0, 1.0),
		vec3(0.5, 0.7, 1.0),
		t
	);
}

// Intersection between a ray and a triangle
Intersection intersect_triangle(Ray ray, uint index)
{
	// Create sphere from object data
	vec4 prop = objects.data[index];
	vec3 v1 = objects.data[index + 1].xyz;
	vec3 v2 = objects.data[index + 2].xyz;
	vec3 v3 = objects.data[index + 3].xyz;

	Triangle triangle = Triangle(v1, v2, v3);

	// Intersect ray with sphere
	Intersection intersection = intersect_shape(ray, triangle);

	// If intersection is valid, compute color
	if (intersection.time > 0.0) {
		// Get material index at the second element
		uint mati = floatBitsToUint(prop.y);

		// Get material from the materials buffer
		intersection.color = materials.data[mati].xyz;
		intersection.shading = materials.data[mati].w;
	}

	return intersection;
}

// Intersection between a ray and a sphere
Intersection intersect_sphere(Ray ray, uint index)
{
	// Create sphere from object data
	vec4 prop = objects.data[index];
	vec4 data = objects.data[index + 1];

	vec3 center = data.xyz;
	float radius = data.w;

	Sphere sphere = Sphere(center, radius);

	// Intersect ray with sphere
	Intersection intersection = intersect_shape(ray, sphere);

	// If intersection is valid, compute color
	if (intersection.time > 0.0) {
		// Get material index at the second element
		uint mati = floatBitsToUint(prop.y);

		// Get material from the materials buffer
		intersection.color = materials.data[mati].xyz;
		intersection.shading = materials.data[mati].w;
	}

	return intersection;
}

// Get object intersection
Intersection intersect(Ray ray, uint index)
{
	// Check type of the object
	float id = objects.data[index].x;

	// TODO: some smarter way to do this
	if (id == OBJECT_TYPE_TRIANGLE)
		return intersect_triangle(ray, index);
	if (id == OBJECT_TYPE_SPHERE)
		return intersect_sphere(ray, index);
	
	return Intersection(-1.0, vec3(0.0), vec3(0.0), SHADING_TYPE_NONE);
}

// Get index of cloests object
Hit closest_object(Ray ray)
{
	int min_index = -1;
	
	// Starting time
	Intersection mini = Intersection(
		1.0/0.0, vec3(0.0), vec3(0.0),
		SHADING_TYPE_NONE
	);

	for (int i = 0; i < world.objects; i++) {
		uint index = world.indices[i];
		Intersection it = intersect(ray, index);

		// Intersection it = intersect_shape(ray, sphere);
		if (it.time > 0.0 && it.time < mini.time) {
			min_index = i;
			mini = it;
		}
	}

	// Color of closest object
	vec3 color = background(ray);
	if (min_index >= 0)
		color = mini.color;

	vec3 point = ray.origin + ray.direction * mini.time;

	return Hit(min_index, mini.time, point, mini.normal, color, mini.shading);
}

vec3 color_at(Ray ray)
{
	// Shadow bias
	// TODO: why is this so large?
	float bias = 0.1;

	// Maximum recursion depth
	int max_depth = 2;
	
	Hit hit = closest_object(ray);
	
	vec3 color = hit.color;
	if (hit.object != -1) {
		// Calculate diffuse lighting
		// TODO: account all lights, after adding flat shading
		vec3 light_pos = lights.data[0].yzw;
		vec3 light_dir = normalize(light_pos - hit.point);

		// Calculate shadow factor by tracing path to light
		float shadow = 0.0;

		// Shadow origin, taking bias into account
		vec3 shadow_origin = hit.point + hit.normal * bias;

		Ray reverse = Ray(shadow_origin, light_dir);
		Hit hit_light = closest_object(reverse);

		// Check the shading type (lighting....)
		// TODO: clean
		if (hit.shading == SHADING_TYPE_LIGHT) {
			return color;
		} else if (hit.shading == SHADING_TYPE_FLAT) {
			// Still need to account for shadows
			// TODO: method to check for valid shadow hit

			if (hit_light.object >= 0 && hit_light.shading != SHADING_TYPE_LIGHT) {
				shadow = 1.0;
			}

			// TODO: ambience variables
			color *= clamp(1.0 - 0.9 * shadow + 0.15, 0.0, 1.0);
			return color;
		}

		// Diffuse lighting
		float diff = max(dot(light_dir, hit.normal), 0.0);

		// Specular lighting
		vec3 view_dir = normalize(world.camera.xyz - hit.point);
		vec3 reflect_dir = reflect(-light_dir, hit.normal);
		float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0);

		int obj = hit_light.object;
		if (obj >= 0 && hit_light.shading != SHADING_TYPE_LIGHT) {
			// Fraction of lights path
			shadow = 1.0;
			spec = 0.0;
		}

		// TODO: different light/shading modes, using ImGui
		color = hit.color * clamp(spec + diff * (1.0 - 0.9 * shadow) + 0.15, 0.0, 1.0);
		// color = discretize(color, 4);
		return color;
	}

	return color;
}

#define MAX_SAMPLES 16

// Sample offsets
// 	Every subarray of a power of 2
//	should be a valid sampling space
const vec2 offsets[MAX_SAMPLES] = {
	vec2(-0.94201624,	-0.39906216),
	vec2(0.94558609,	-0.76890725),
	vec2(-0.094184101,	-0.92938870),
	vec2(0.34495938,	0.29387760),
	vec2(-0.51947840,	0.84840893),
	vec2(0.15452093,	0.60745321),
	vec2(-0.86442297,	-0.085755840),
	vec2(0.63357930,	-0.62802040),
	vec2(0.72695800,	0.43725902),
	vec2(0.79011270,	0.10122100),
	vec2(-0.96886787,	-0.11038005),
	vec2(-0.17360089,	-0.93959504),
	vec2(0.68785068,	-0.72534355),
	vec2(0.98994950,	0.14948041),
	vec2(-0.031639270,	0.99946570),
	vec2(-0.79587721,	0.60407257)
};

// TODO: pass as world parameter
#define SAMPLES 1

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	// Return if out of bounds
	if (y0 >= world.height || x0 >= world.width)
		return;

	uint ysize = gl_NumWorkGroups.y;
	uint xsize = gl_NumWorkGroups.x;

	vec2 dimensions = vec2(world.width, world.height);
	/* for (uint y = y0; y < world.height; y += ysize) {
		for (uint x = x0; x < world.width; x += xsize) {
			uint index = y * world.width + x;
			
			vec3 sum = vec3(0.0);
			for (uint s = 0; s < SAMPLES; s++) {
				// Sample ray
				vec2 point = vec2(x + 0.5, y + 0.5) + offsets[s];

				vec2 uv = point / dimensions;
				Ray ray = make_ray(
					uv,
					world.camera.xyz,
					world.cforward.xyz, world.cup.xyz, world.cright.xyz,
					world.tunings.y,
					world.tunings.z
				);

				// Get color
				vec3 color = color_at(ray);

				// Accumulate color
				sum += color;
			}

			// Average samples
			sum /= float(SAMPLES);

			// Set pixel color
			frame.pixels[index] = cast_color(sum);
		}
	} */
			
	uint index = y0 * world.width + x0;
		
	// Sample ray
	// TODO: use z index for sample offset index
	vec2 point = vec2(x0 + 0.5, y0 + 0.5) + offsets[0];

	vec2 uv = point / dimensions;
	Ray ray = make_ray(
		uv,
		world.camera.xyz,
		world.cforward.xyz, world.cup.xyz, world.cright.xyz,
		world.tunings.y,
		world.tunings.z
	);

	// Get color
	vec3 color = color_at(ray);

	// Set pixel color
	frame.pixels[index] = cast_color(color);
}
