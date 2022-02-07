#version 430

#include "ray.glsl"
#include "shapes.glsl"
#include "intersect.glsl"
#include "color.glsl"

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
	vec3 cforward;
	vec3 cup;
	vec3 cright;

	// TODO: make a camera structure
	// plus transform
	
	// fov, scale, aspect
	float fov;
	float scale;
	float aspect;

	// TODO: indices for objects, lights, background
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

// Closest object information
struct Hit {
	int	object;
	float	time;
	vec3	point;
	vec3	normal;
	vec3	color;
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

// Get index of cloests object
Hit closest_object(Ray ray)
{
	int min_index = -1;
	
	// Starting time
	Intersection mini = Intersection(1.0/0.0, vec3(0.0));

	// Data index
	uint di = 0;

	for (int i = 0; i < world.objects; i++) {
		vec3 p = objects.data[di].xyz;
		float r = objects.data[di].w;

		Sphere sphere = Sphere(p, r);

		Intersection it = intersects(sphere, ray);
		if (it.time > 0 && it.time < mini.time) {
			min_index = i;
			mini = it;
		}

		di += 2;
	}

	// Color of closest object
	vec3 color = background(ray);
	if (min_index >= 0)
		color = objects.data[min_index * 2 + 1].xyz;

	vec3 point = ray.origin + ray.direction * mini.time;

	return Hit(min_index, mini.time, point, mini.normal, color);
}

vec3 color_at(Ray ray)
{
	// Shadow bias
	// TODO: why is this so large?
	float bias = 0.1;

	// Maximum recursion depth
	int max_depth = 2;
	
	// TODO: pass color as vec3
	// vec3 color = cast_color(world.background);
	Hit hit = closest_object(ray);
	
	vec3 color = hit.color;
	if (hit.object != -1) {
		color = hit.color;

		// Calculate diffuse lighting
		vec3 light_pos = lights.data[0].xyz;
		vec3 light_dir = normalize(light_pos - hit.point);

		// Calculate shadow factor by tracing path to light
		float shadow = 0.0;

		// Shadow origin, taking bias into account
		vec3 shadow_origin = hit.point + hit.normal * bias;

		Ray reverse = Ray(shadow_origin, light_dir);
		Hit hit_light = closest_object(reverse);

		int obj = hit_light.object;
		if (obj >= 0) {
			// Fraction of lights path
			shadow = 1.0;
		}

		// Diffuse lighting
		float diff = max(dot(light_dir, hit.normal), 0.0);

		// Specular lighting
		vec3 view_dir = normalize(world.camera - hit.point);
		vec3 reflect_dir = reflect(-light_dir, hit.normal);
		float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0);

		// TODO: different light/shading modes, using ImGui
		color = hit.color * clamp((spec + diff) * (1.0 - 0.9 * shadow) + 0.15, 0.0, 1.0);

		// color = vec3(1.0 - shadow);
		// color = discretize(color, 16.0f);
		// color = hit.normal;
	}

	/* int iter = 0;

	vec3 light_pos = lights.data[0].xyz;

	// Send the first ray
	Hit hit = closest_object(ray);

	color = vec3(0.0);
	while (hit.object != -1 && iter < max_depth) {
		// Get contribution from light
		vec3 light_dir = normalize(light_pos - hit.point);
		float diff = max(dot(light_dir, hit.normal), 0.0);

		// Get contribution from reflection
		vec3 view_dir = normalize(world.camera - hit.point);
		vec3 reflect_dir = reflect(-light_dir, hit.normal);

		float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0);

		// Calculate shadow factor by tracing path to light
		float shadow = 0.0;

		// Shadow origin, taking bias into account
		vec3 shadow_origin = hit.point + hit.normal * bias;

		Ray reverse = Ray(shadow_origin, light_dir);
		Hit hit_light = closest_object(reverse);

		int obj = hit_light.object;
		if (obj >= 0) {
			// Fraction of lights path
			shadow = 1.0;
		}

		// Combine contributions
		color += hit.color * clamp((spec + diff) * (1.0 - 0.9 * shadow) + 0.15, 0.0, 1.0);

		// Send reflection ray
		vec3 reflect_origin = hit.point + hit.normal * bias;
		Ray reflect_ray = Ray(reflect_origin, reflect_dir);

		// Get next closest object
		hit = closest_object(reflect_ray);

		iter++;
	}

	// If no object was hit, return background color
	if (iter == 0)
		color = cast_color(world.background); */

	return color;
}

#define MAX_SAMPLES 16

// Sample offsets
// 	Every subarray of a power of 2
//	should be a valid sampling space
const vec2 offsets[MAX_SAMPLES] = {
	vec2(-0.94201624, -0.39906216),
	vec2(0.94558609, -0.76890725),
	vec2(-0.094184101, -0.92938870),
	vec2(0.34495938, 0.29387760),
	vec2(-0.51947840, 0.84840893),
	vec2(0.15452093, 0.60745321),
	vec2(-0.86442297, -0.085755840),
	vec2(0.63357930, -0.62802040),
	vec2(0.72695800, 0.43725902),
	vec2(0.79011270, 0.10122100),
	vec2(-0.96886787, -0.11038005),
	vec2(-0.17360089, -0.93959504),
	vec2(0.68785068, -0.72534355),
	vec2(0.98994950, 0.14948041),
	vec2(-0.031639270, 0.99946570),
	vec2(-0.79587721, 0.60407257)
};

// TODO: pass as world parameter
#define SAMPLES 6

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	uint ysize = gl_NumWorkGroups.y;
	uint xsize = gl_NumWorkGroups.x;

	// TODO: split into functions
	vec2 dimensions = vec2(world.width, world.height);
	for (uint y = y0; y < world.height; y += ysize) {
		for (uint x = x0; x < world.width; x += xsize) {
			uint index = y * world.width + x;
			
			vec3 sum = vec3(0.0);
			for (uint s = 0; s < SAMPLES; s++) {
				// Sample ray
				vec2 point = vec2(x + 0.5, y + 0.5) + offsets[s];

				vec2 uv = point / dimensions;
				Ray ray = make_ray(
					uv,
					world.camera,
					world.cforward, world.cup, world.cright,
					world.scale, world.aspect
				);

				// Get color
				vec3 color = color_at(ray);

				// Accumulate color
				sum += color;
			}

			/*vec2 uv = vec2(x + 0.5, y + 0.5)
				/ vec2(world.width, world.height);

			// Create ray
			Ray ray = make_ray(
				uv,
				world.camera,
				world.cforward, world.cup, world.cright,
				world.scale, world.aspect
			);
		
			// Get color at ray
			vec3 color = color_at(ray); */

			// Average samples
			sum /= float(SAMPLES);

			// Set pixel color
			frame.pixels[index] = cast_color(sum);
		}
	}
}
