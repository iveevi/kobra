#version 450

// Import bindings
#include "mesh_bindings.h"

// TODO: all these buffer bindings should be in a single file
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

// Mesh transforms
// TODO: is this even needed? its too slow to compute every frame
layout (set = 0, binding = MESH_BINDING_TRANSFORMS, std430) buffer Transforms
{
	mat4 data[];
} transforms;

// Acceleration structure
layout (set = 0, binding = MESH_BINDING_BVH, std430) buffer BVH
{
	vec4 data[];
} bvh;

// Materials
layout (set = 0, binding = MESH_BINDING_MATERIALS, std430) buffer Materials
{
	vec4 data[];
} materials;

// Push constants
// TODO: # of samples should be a push constant
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
};

// Default "constructor"
Material mat_default()
{
	return Material(
		vec3(0.5f, 0.5f, 0.5f)
	);
}

// Convert raw material at index
Material mat_at(uint index)
{
	vec4 raw0 = materials.data[2 * index];
	vec4 raw1 = materials.data[2 * index + 1];

	return Material(raw0.xyz);
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
	float id = triangles.data[index].w;

	uint a = floatBitsToUint(ia);
	uint b = floatBitsToUint(ib);
	uint c = floatBitsToUint(ic);
	uint d = floatBitsToUint(id);

	vec3 v1 = vertices.data[a].xyz;
	vec3 v2 = vertices.data[b].xyz;
	vec3 v3 = vertices.data[c].xyz;

	Triangle triangle = Triangle(v1, v2, v3);

	// Get intersection
	Intersection it = intersect_shape(ray, triangle);

	// If intersection is valid, compute material
	if (it.time > 0.0) {
		// Get material index at the second element
		it.mat = mat_at(d);
	}

	// Intersect ray with sphere
	return it;
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
// Closest object information
struct Hit {
	int	object;
	float	time;
	vec3	point;
	vec3	normal;

	Material mat;
};

// Get closest object
Hit closest_object(Ray ray)
{
	int min_index = -1;

	// Starting intersection
	Intersection mini = Intersection(
		1.0/0.0, vec3(0.0),
		mat_default()
	);

	// Traverse BVH as a threaded binary tree
	int node = 0;
	while (node != -1) {
		if (leaf(node)) {
			// Get object index
			int index = object(node);

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

	// Color of closest object
	vec3 color = vec3(0.0);	// TODO: sample from either texture or gradient
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

// TODO: light transport header
vec3 light = vec3(0.0, 5.0, 3.0);

vec3 color_at(Ray ray)
{
	// Bias for intersection
	float bias = 0.1;

	Hit hit = closest_object(ray);
	
	if (hit.object != -1) {
		// Calcula basic diffuse lighting
		vec3 light_dir = normalize(light - hit.point);
		float diffuse = max(dot(hit.normal, light_dir), 0.0);
		return hit.mat.albedo * diffuse;
	}

	return hit.mat.albedo;
}

// TODO: color wheel (if needed) goes in a header

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

	// Hit h = closest_object(ray);

	// Light transport
	vec3 color = color_at(ray);

	// Get index
	uint index = y0 * viewport.width + x0;
	frame.pixels[index] = cast_color(color);
}
