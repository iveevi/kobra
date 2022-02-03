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
} world;

layout (set = 0, binding = 2, std430) buffer Objects
{
	// Object layout:
	// vec4 descriptor (sphere, plane, mesh, etc. + size and offset)
	// vec4 color
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
	vec3 color = objects.data[min_index * 2 + 1].xyz;
	vec3 point = ray.origin + ray.direction * mini.time;

	return Hit(min_index, mini.time, point, mini.normal, color);
}

vec3 color_at(Ray ray)
{
	// Shadow bias
	// TODO: why is this so large?
	float bias = 0.1;

	Hit hit = closest_object(ray);
	
	// TODO: pass color as vec3
	vec3 color = cast_color(world.background);
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

		vec3 c = objects.data[hit.object * 2 + 1].xyz;
		float diff = max(dot(light_dir, hit.normal), 0.0);

		// TODO: different light/shading modes, using ImGui
		color = c * clamp(diff * (1.0 - 0.9 * shadow) + 0.15, 0.0, 1.0);

		// color = vec3(1.0 - shadow);
		// color = discretize(color, 16.0f);
	}

	return color;
}

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	uint ysize = gl_NumWorkGroups.y;
	uint xsize = gl_NumWorkGroups.x;

	for (uint y = y0; y < world.height; y += ysize) {
		for (uint x = x0; x < world.width; x += xsize) {
			uint index = y * world.width + x;

			vec2 uv = vec2(x + 0.5, y + 0.5)
				/ vec2(world.width, world.height);

			// Create ray
			Ray ray = make_ray(
				uv,
				world.camera,
				world.cforward, world.cup, world.cright,
				world.scale, world.aspect
			);
		
			// Get color at ray
			vec3 color = color_at(ray);
			frame.pixels[index] = cast_color(color);
		}
	}
}
