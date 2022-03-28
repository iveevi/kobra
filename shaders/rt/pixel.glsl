#version 430

// TODO: rename file

#include "../../include/types.hpp"

#include "common/ray.glsl"
#include "common/intersect.glsl"
#include "common/color.glsl"
#include "common/bbox.glsl"

// TODO: replace version with command line argument

// TODO: move layouts to separate file
layout (set = 0, binding = 0, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = 1, std430) buffer World
{
	uint objects;		// TODO: this is useless
	uint primitives;
	uint lights;

	uint width;
	uint height;

	uint options;
	int discretize;

	vec4 camera;
	vec4 cforward;
	vec4 cup;
	vec4 cright;

	vec4 tunings;

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
// TODO: move to separate file
layout (set = 0, binding = 4, std430) buffer Materials
{
	// Material layout:
	// vec4 descriptor (color[3], shading[1])
	// vec4 properties (reflectance[1], refractance[1])
	// note that refractance is index of refraction
	vec4 data[];
} materials;

// Create material from index
Material material_for(uint mati)
{
	vec4 r1 = materials.data[mati];
	vec4 r2 = materials.data[mati + 1];

	return Material(
		vec3(r1.x, r1.y, r1.z), r1.w,
		r2.x, r2.y, vec2(r2.z, r2.w)
	);
}

// BVH
layout (set = 0, binding = 5, std430) buffer BVH
{
	// BVH layout:
	// vec4 header
	// vec4 centroid
	// vec4 dimenions
	vec4 data[];
} bvh;

// Stack (for BVH traversal)
layout (set = 0, binding = 6, std430) buffer Stack
{
	int data[];
} stack;

// Debug buffer
layout (set = 0, binding = 7, std430) buffer Debug
{
	vec4 data[];
} debug;

// Vertex buffer
layout (set = 0, binding = 8, std430) buffer Vertices
{
	// For now is just a vec3 position
	vec4 data[];
} vertices;

// Transforms buffer
layout (set = 0, binding = 9, std430) buffer Transforms
{
	// Transform as a model matrix
	mat4 data[];
} transforms;

// Texture module
#include "common/textures.glsl"

// Closest object information
// TODO: this should be obslete
struct Hit {
	int	object;
	float	time;
	vec3	point;
	vec3	normal;

	Material mat;
};

// Background color corresponding to ray
// corresponds to a sky like gradient
vec3 background(Ray ray)
{
	// Sample sky texture
	vec3 dir = normalize(ray.direction);
	return sample_texture(dir, 0).xyz;
}

// Intersection between a ray and a triangle
Intersection intersect_triangle(Ray ray, uint index)
{
	// Create sphere from object data
	vec4 prop = objects.data[index];
	vec4 indices = objects.data[index + 1];

	// Transform index
	uint tati = floatBitsToUint(prop.z);
	mat4 model = transforms.data[tati];

	uint a = floatBitsToUint(indices.x);
	uint b = floatBitsToUint(indices.y);
	uint c = floatBitsToUint(indices.z);

	vec3 v1 = vec3(model * vertices.data[a]);
	vec3 v2 = vec3(model * vertices.data[b]);
	vec3 v3 = vec3(model * vertices.data[c]);

	Triangle triangle = Triangle(v1, v2, v3);

	// Intersect ray with sphere
	Intersection intersection = intersect_shape(ray, triangle);

	// If intersection is valid, compute color
	if (intersection.time >= 0.0) {
		// Get material index at the second element
		uint mati = floatBitsToUint(prop.y);
		intersection.mat = material_for(mati);
	}

	return intersection;
}

// Intersection between a ray and a sphere
Intersection intersect_sphere(Ray ray, uint index)
{
	// Create sphere from object data
	vec4 prop = objects.data[index];
	vec4 data = objects.data[index + 1];

	// Transform index
	uint tati = floatBitsToUint(prop.z);
	mat4 model = transforms.data[tati];

	vec3 center = vec3(model * vec4(0.0, 0.0, 0.0, 1.0));
	float radius = data.x;

	Sphere sphere = Sphere(center, radius);

	// Intersect ray with sphere
	Intersection intersection = intersect_shape(ray, sphere);

	// If intersection is valid, compute color
	if (intersection.time > 0.0) {
		// Get material index at the second element
		uint mati = floatBitsToUint(prop.y);
		intersection.mat = material_for(mati);
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

	return Intersection(-1.0, vec3(0.0), mat_default());
}

// TODO: put all bvh functions in separate file

// Get left and right child of the node
int hit(int node)
{
	vec4 prop = bvh.data[node];
	return floatBitsToInt(prop.z);
}

int miss(int node)
{
	vec4 prop = bvh.data[node];
	return floatBitsToInt(prop.w);
}

int object(int node)
{
	vec4 prop = bvh.data[node];
	return floatBitsToInt(prop.y);
}

bool leaf(int node)
{
	vec4 prop = bvh.data[node];
	return prop.x == 0x1;
}

BoundingBox bbox(int node)
{
	vec3 min = bvh.data[node + 1].xyz;
	vec3 max = bvh.data[node + 2].xyz;

	return BoundingBox(min, max);
}

// Get index of cloests object
int min_index = -1;

Hit closest_object(Ray ray)
{
	min_index = -1;

	// Starting time
	Intersection mini = Intersection(
		1.0/0.0, vec3(0.0),
		mat_default()
	);

	// If the ray is null
	if (ray.direction == vec3(0.0))
		return Hit(min_index, 0.0, vec3(0.0), vec3(0.0), mat_default());

#if 1

	// Traverse BVH as a threaded binary tree
	int node = 0;
	int count = 0;
	while (node != -1) {
		count++;
		if (leaf(node)) {
			// Get object index
			// int iobj = floatBitsToInt(bvh.data[node].y);
			int iobj = object(node);
			uint index = world.indices[iobj];

			// Get object
			Intersection it = intersect(ray, index);

			// If intersection is valid, update minimum
			if (it.time > 0.0 && it.time < mini.time) {
				min_index = iobj;
				mini = it;
			}

			// Go to next node (ame as miss)
			node = miss(node);
		} else {
			// Get bounding box
			BoundingBox box = bbox(node);

			// Check if ray intersects (or is inside)
			// the bounding box
			float t = intersect_box(ray, box);
			bool inside = in_box(ray.origin, box);

			if (t > 0.0 || inside) {
				// Traverse left child
				node = hit(node);
			} else {
				// Traverse right child
				node = miss(node);
			}
		}
	}

#else

	// Traverse linearly
	int count = 0;
	for (int i = 0; i < world.primitives; i++) {
		count++;
		// Get object index
		uint index = world.indices[i];

		// Get object
		Intersection it = intersect(ray, index);

		// If intersection is valid, update minimum
		if (it.time > 0.0 && it.time < mini.time) {
			min_index = i;
			mini = it;
		}
	}

#endif

	// Color of closest object
	vec3 color = background(ray);
	if (min_index < 0)
		mini.mat.albedo = color;

	vec3 point = ray.origin + ray.direction * mini.time;

	return Hit(
		min_index,
		mini.time,
		point,
		mini.normal,
		mini.mat
	);
}

// "Phong" calculation
vec3 color_calc(Hit hit, Ray ray)
{
	// If no hit, just return background color
	if (hit.object < 0)
		return hit.mat.albedo;

	// Check for light type again
	if (hit.mat.shading == SHADING_TYPE_LIGHT)
		return hit.mat.albedo;

	// Intersection bias
	float bias = 0.1;

	// Light position (fixed for now)
	vec3 light_position = lights.data[0].yzw; // TODO: function to get light position
	vec3 light_direction = normalize(light_position - hit.point);

	// Shadow calculation
	vec3 shadow_origin = hit.point + hit.normal * bias;

	Ray shadow_ray = Ray(
		shadow_origin,
		light_direction
	);

	Hit shadow_hit = closest_object(shadow_ray);

	float shadow = 0.0;
	if (shadow_hit.object >= 0 && shadow_hit.mat.shading != SHADING_TYPE_LIGHT)
		shadow = 1.0;

	// Diffuse
	float diffuse = max(dot(hit.normal, light_direction), 0.0);

	// Specular
	vec3 view_direction = normalize(hit.point - ray.origin);
	vec3 reflect_direction = reflect(light_direction, hit.normal);
	float specular = pow(
		max(dot(view_direction, reflect_direction), 0.0),
		hit.mat.specular
	);

	specular = 0.0;

	// Combine diffuse and shadow
	float ambience = 0.15;

	float diffuse_specular = ambience + (1.0 - ambience) * (diffuse + specular);
	float factor = clamp(
		diffuse_specular * (1.0 - 0.9 * shadow),
		0.0, 1.0
	);

	// Return final color
	return hit.mat.albedo * factor;
}

vec3 color_calc_flat(Hit hit)
{
	// Intersection bias
	float bias = 0.1;

	// Light position (fixed for now)
	vec3 light_position = lights.data[0].yzw;
	vec3 light_direction = normalize(light_position - hit.point);

	// Shadow calculation
	vec3 shadow_origin = hit.point + hit.normal * bias;

	Ray shadow_ray = Ray(
		shadow_origin,
		light_direction
	);

	Hit shadow_hit = closest_object(shadow_ray);

	float shadow = 0.0;
	if (shadow_hit.object >= 0 && shadow_hit.mat.shading != SHADING_TYPE_LIGHT)
		shadow = 1.0;

	// Calculate factor
	float factor = clamp(1.0 - 0.9 * shadow, 0.0, 1.0);

	// Return final color
	return hit.mat.albedo * factor;
}

struct Branch {
	Ray	refl;
	float	erefl;

	Ray	refr;
	float	erefr;

	float	prefr;
};

Branch null_branch()
{
	return Branch(
		Ray(vec3(0.0), vec3(0.0)),
		0.0,
		Ray(vec3(0.0), vec3(0.0)),
		0.0,
		0.0
	);
}

Branch branch(Hit hit, Ray ray, float prefr)
{
	// If the ray is null
	if (ray.direction == vec3(0.0))
		return null_branch();

	// Intersection bias
	float bias = 0.1;

	// New origins
	vec3 refl_origin = hit.point + hit.normal * bias;
	vec3 refr_origin = hit.point - hit.normal * bias;

	// New directions
	float eta = prefr/hit.mat.ior.x;
	vec3 refl_direction = reflect(ray.direction, hit.normal);
	vec3 refr_direction = refract(ray.direction, hit.normal, hit.mat.ior.x);

	// Return reflection structure
	// TODO: fresnel calculation for energy
	return Branch(
		Ray(refl_origin, refl_direction),
		hit.mat.ior.x,
		Ray(refr_origin, refr_direction),
		1 - hit.mat.ior.x,
		hit.mat.ior.x
	);
}

#include "common/transport.glsl"

vec3 color_at(Ray ray)
{
	// Bias for intersection
	float bias = 0.1;

	Hit hit = closest_object(ray);
	if (hit.object != -1) {
		if (hit.mat.shading == SHADING_TYPE_LIGHT) {
			return hit.mat.albedo;
		} else if (hit.mat.shading == SHADING_TYPE_FLAT) {
			return color_calc_flat(hit);
		}

		// Apply 1/2/3 bounces
		Ray r1 = ray;
		Branch b1 = null_branch(),
			b2 = null_branch(),
			b3 = null_branch();

		Hit h1 = hit;
		b1 = branch(h1, r1, h1.mat.ior.x);

		Hit h2 = closest_object(b1.refl);
		Hit h3 = closest_object(b1.refr);

		// Get individual colors
		vec3 c1 = diffuse(h1, r1);
		vec3 c2 = diffuse(h2, b1.refl);
		vec3 c3 = diffuse(h3, b1.refr);

		return clamp(c1 + b1.erefl * c2 + b1.erefr * c3, 0, 1);
	}

	return hit.mat.albedo;
}

// TODO: pass samples as world parameter

// Generate sampling noise
float rand(vec2 co)
{
	return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 pnoise = vec2(1.0, 0.0);
vec2 unit_noise()
{
	float f1 = rand(pnoise);
	float f2 = rand(pnoise + f1);
	pnoise = normalize(vec2(f1, f2));
	return pnoise;
}

#define COLOR_WHEEL_SIZE 8

// Bounding box color wheel
const vec3 colors[COLOR_WHEEL_SIZE] = {
	vec3(1.0, 0.0, 0.0),
	vec3(1.0, 1.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 1.0, 1.0),
	vec3(1.0, 0.0, 1.0),
	vec3(0.0, 0.0, 1.0),
	vec3(1.0, 0.5, 0.0),
	vec3(0.5, 0.0, 1.0)
};

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

	uint index = y0 * world.width + x0;

	// Sample ray
	// TODO: use z index for sample offset index
	vec3 sum = vec3(0.0);
	vec2 point = vec2(x0 + 0.5, y0 + 0.5);

	float n = 1.0;
	for (int i = 0; i < n; i++) {
		vec2 uv = (point + unit_noise()/2) / dimensions;
		Ray ray = make_ray(
			uv,
			world.camera.xyz,
			world.cforward.xyz, world.cup.xyz, world.cright.xyz,
			world.tunings.y,
			world.tunings.z
		);

		// Get color
		vec3 color = color_at(ray);

		sum += color;
	}

	vec3 color = sum/n;

	// Descretize color if needed
	if (world.discretize > 0)
		color = discretize_grey(color, world.discretize);

	// TODO: constants for mask
	if ((world.options & 0x1) == 0x1) {
		Ray ray = make_ray(
			vec2(x0 + 0.5, y0 + 0.5) / dimensions,
			world.camera.xyz,
			world.cforward.xyz, world.cup.xyz, world.cright.xyz,
			world.tunings.y,
			world.tunings.z
		);

		// Iterate through BVHs
		int node = 0;
		while (node != -1) {
			if (leaf(node)) {
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
					vec3 blend = colors[node % COLOR_WHEEL_SIZE];
					color = mix(color, blend, 0.15);
					node = hit(node);
				} else {
					// Missed
					node = miss(node);
				}
			}
		}
	}

	// Set pixel color
	frame.pixels[index] = cast_color(color);
}
