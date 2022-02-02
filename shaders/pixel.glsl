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

	// TODO: should later contain color
	
	// TODO:remove this
	vec3 normal;
};

// Get index of cloests object
Hit closest_object(Ray ray)
{
	int min_index = -1;
	float min_time = 1.0/0.0;
	vec3 normal = vec3(0.0);

	// Data index
	uint di = 0;

	for (int i = 0; i < world.objects; i++) {
		vec3 p = objects.data[di].xyz;
		float r = objects.data[di].w;

		Sphere sphere = Sphere(p, r);

		Intersection it = intersects(sphere, ray);
		if (it.time > 0 && it.time < min_time) {
			min_time = it.time;
			min_index = i;
			normal = it.normal;
		}

		di++;
	}

	// Hit point using ray
	vec3 point = ray.origin + ray.direction * min_time;

	return Hit(min_index, min_time, point, normal);
}

// Uint color to vec3 color
vec3 cast_color(uint c)
{
	return vec3(
		float(c & 0xFF) / 255.0,
		float((c >> 8) & 0xFF) / 255.0,
		float((c >> 16) & 0xFF) / 255.0
	);
}

// Vec3 color to uint color
uint cast_color(vec3 c)
{
	return uint(c.x * 255.0)
		| (uint(c.y * 255.0) << 8)
		| (uint(c.z * 255.0) << 16);
}

void main()
{
	// Calculate initial x and y and offset indexes
	uint sphere_color = 0x00FF00;

	uint colors[] = {
		0xFF0000,
		0x00FF00,
		0x0000FF,
		0xFFFF00,
		0x00FFFF,
		0xFF00FF,
		0xFFFFFF
	};

	uint color;

	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	uint ysize = gl_NumWorkGroups.y;
	uint xsize = gl_NumWorkGroups.x;

	vec3 light = lights.data[0].xyz;

	for (uint y = y0; y < world.height; y += ysize) {
		for (uint x = x0; x < world.width; x += xsize) {
			uint index = y * world.width + x;

			vec2 uv = vec2(x + 0.5, y + 0.5) / vec2(world.width, world.height);

			Ray ray = make_ray(uv, world.camera, world.scale, world.aspect);
			
			Hit hit = closest_object(ray);
			if (hit.object >= 0) {
				color = colors[hit.object % 7];

				// Calculate diffuse
				vec3 light_dir = normalize(light - hit.point);
				float diffuse = max(dot(hit.normal, light_dir), 0.0);

				vec3 c = cast_color(color);
				c *= diffuse;

				color = cast_color(c);
			} else {
				color = world.background;
			}

			// TODO: convert color from vec3 to uint
			frame.pixels[index] = color;
		}
	}
}
