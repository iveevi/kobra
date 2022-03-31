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

layout (set = 0, binding = MESH_BINDING_BVH, std430) buffer BVH
{
	vec4 data[];
} bvh;

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
#include "common/bbox.glsl"

// Triangle primitive
struct Triangle {
	vec3 v1;
	vec3 v2;
	vec3 v3;
};

// Material structure
struct Material {
	vec3 albedo;
	float shading;

	float specular;
	float reflectance;
	
	// Index of refraction as a complex number
	vec2 ior;
};

// Default "constructor"
Material mat_default()
{
	return Material(
		vec3(0.5f, 0.5f, 0.5f), 0.0f,
		0.0f, 0.0f, vec2(0.0f, 0.0f)
	);
}

// TODO: intersection header
// Intersection
struct Intersection {
	float	time;
	vec3	normal;

	Material mat;
};

float _intersect_t(Triangle t, Ray r)
{
	vec3 e1 = t.v2 - t.v1;
	vec3 e2 = t.v3 - t.v1;
	vec3 s1 = cross(r.direction, e2);
	float divisor = dot(s1, e1);
	if (divisor == 0.0)
		return -1.0;
	vec3 s = r.origin - t.v1;
	float inv_divisor = 1.0 / divisor;
	float b1 = dot(s, s1) * inv_divisor;
	if (b1 < 0.0 || b1 > 1.0)
		return -1.0;
	vec3 s2 = cross(s, e1);
	float b2 = dot(r.direction, s2) * inv_divisor;
	if (b2 < 0.0 || b1 + b2 > 1.0)
		return -1.0;
	float time = dot(e2, s2) * inv_divisor;
	return time;
}

Intersection intersect_shape(Ray r, Triangle t)
{
	// Get intersection time
	float time = _intersect_t(t, r);
	vec3 n = vec3(0.0);

	if (time < 0.0)
		return Intersection(-1.0, n, mat_default());

	// Calculate the normal
	vec3 e1 = t.v2 - t.v1;
	vec3 e2 = t.v3 - t.v1;
	n = cross(e1, e2);
	n = normalize(n);

	// Negate normal if in the same direction as the ray
	if (dot(n, r.direction) > 0.0)
		n = -n;

	return Intersection(time, n, mat_default());
}

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

// TODO: bvh header
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

// Get closest object
float closest_object(Ray ray)
{
	int min_index = -1;

	// Starting time
	Intersection mini = Intersection(
		1.0/0.0, vec3(0.0),
		mat_default()
	);

	// Traverse BVH as a threaded binary tree
	int node = 0;
	int count = 0;
	while (node != -1) {
		count++;
		if (leaf(node)) {
			// return 1.0;
			// Get object index
			// int iobj = floatBitsToInt(bvh.data[node].y);
			int index = object(node);

			// uint index = world.indices[iobj];

			// Get object
			Intersection it = ray_intersect(ray, index);

			// If intersection is valid, update minimum
			if (it.time > 0.0 && it.time < mini.time) {
				min_index = index;
				mini = it;
			}

			// Go to next node (ame as miss)
			node = miss(node);
		} else {
			// return 1.0;
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
		color = vec3(1, 0, 1);
	
	// Get index
	uint index = y0 * viewport.width + x0;
	frame.pixels[index] = cast_color(color);
}
