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

// Lights
layout (set = 0, binding = MESH_BINDING_LIGHTS, std430) buffer Lights
{
	vec4 data[];
} lights;

// Light indices
layout (set = 0, binding = MESH_BINDING_LIGHT_INDICES, std430) buffer LightIndices
{
	uint data[];
} light_indices;

// Textures
layout (set = 0, binding = MESH_BINDING_ALBEDO) uniform sampler2D s2_albedo;
layout (set = 0, binding = MESH_BINDING_NORMAL_MAPS) uniform sampler2D s2_normals;

// Push constants
// TODO: # of samples should be a push constant
layout (push_constant) uniform PushConstants
{
	// Size variables
	uint	triangles;
	uint	lights;
	uint	samples_per_pixel;
	uint	samples_per_light;

	// Camera
	vec3	camera_position;
	vec3	camera_forward;
	vec3	camera_up;
	vec3	camera_right;

	// scale, aspect
	vec4	properties;
} pc;

// Import other headers
#include "common/color.glsl"
#include "common/ray.glsl"
#include "common/bbox.glsl"
#include "../../include/types.hpp"

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
	float ior;
};

// Default "constructor"
Material mat_default()
{
	return Material(
		vec3(0.5f, 0.5f, 0.5f),
		-1.0f, 1.0f
	);
}

// Convert raw material at index
Material mat_at(uint index)
{
	vec4 raw0 = materials.data[2 * index];
	vec4 raw1 = materials.data[2 * index + 1];

	return Material(
		vec3(raw0.x, raw0.y, raw0.z),
		raw0.w,
		raw1.x
	);
}

// TODO: intersection header
// Intersection
struct Intersection {
	float	time;
	vec3	normal;

	Material mat;
};

float b1;
float b2;

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
	b1 = dot(s, s1) * inv_divisor;
	if (b1 < 0.0 || b1 > 1.0)
		return -1.0;
	vec3 s2 = cross(s, e1);
	b2 = dot(r.direction, s2) * inv_divisor;
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

	// TODO: macro for fixed width vertices
	vec3 v1 = vertices.data[2 * a].xyz;
	vec3 v2 = vertices.data[2 * b].xyz;
	vec3 v3 = vertices.data[2 * c].xyz;

	Triangle triangle = Triangle(v1, v2, v3);

	// Get intersection
	Intersection it = intersect_shape(ray, triangle);

	// If intersection is valid, compute material
	if (it.time > 0.0) {
		// Get material index at the second element
		it.mat = mat_at(d);

		// Get texture coordinates
		vec2 t1 = vertices.data[2 * a + 1].xy;
		vec2 t2 = vertices.data[2 * b + 1].xy;
		vec2 t3 = vertices.data[2 * c + 1].xy;

		// Interpolate texture coordinates
		vec2 tex_coord = t1 * (1 - b1 - b2) + t2 * b1 + t3 * b2;
		tex_coord = clamp(tex_coord, vec2(0.0), vec2(1.0));

		// Get texture
		tex_coord.y = 1.0 - tex_coord.y;
		vec3 tex = texture(s2_normals, tex_coord).xyz;

		// Correct the normal direction
		if (dot(it.normal, tex) < 0.0)
			tex = -tex;

		if (d == 1)
			it.mat.albedo = texture(s2_albedo, tex_coord).xyz;
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
	
// Bias for intersection
float RT_BIAS = 0.1;

// Golden random function
float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio 

float ray_seed = 1.0;

float gold_noise(in vec2 xy, in float seed)
{
	return fract(sin(dot(xy, vec2(12.9898, 78.233))) * seed);
}

// Sample random point in triangle
float u = 0.0;
float v = 0.0;

vec3 sample_triangle(vec3 v1, vec3 v2, vec3 v3, float strata, float i)
{
	// Ignore random-ness if only 1 sample
	if (pc.samples_per_light == 1)
		return (v1 + v2 + v3) / 3.0;

	// Get random point in triangle
	u = gold_noise(vec2(v1), ray_seed);
	v = gold_noise(vec2(v2), ray_seed);

	// In the interval corresponding to the strata and index
	float inv = 1.0 / strata;
	float x = floor(i/strata);
	float y = i - x * strata;

	u = u * inv + x * inv;
	v = v * inv + y * inv;

	// Edge vectors
	vec3 e1 = v2 - v1;
	vec3 e2 = v3 - v1;

	vec3 p = v1 + u * e1 + v * e2;

	// Update seed
	ray_seed = fract((ray_seed + 1.0) * PHI);

	return p;
}

vec3 single_area_light_contr(Hit hit, Ray ray, vec3 v1, vec3 v2, vec3 v3)
{
	vec3 color = vec3(0.0);
	for (int i = 0; i < pc.samples_per_light; i++) {
		// Sample point from triangle
		vec3 light_position = sample_triangle(
			v1, v2, v3,
			sqrt(pc.samples_per_light), i
		);

#if 1
		// Light intensity
		float d = distance(light_position, hit.point);
		float intensity = 1.0/d;

		// Lambertian
		vec3 light_direction = normalize(light_position - hit.point);
		float diffuse = max(dot(light_direction, hit.normal), 0.0);

		// Shadow
		Ray shadow_ray = Ray(
			hit.point + hit.normal * 0.001,
			light_direction
		);

		Hit shadow_hit = closest_object(shadow_ray);
		if (shadow_hit.mat.shading == SHADING_TYPE_EMMISIVE || shadow_hit.time > d) {
			// Add light
			color += hit.mat.albedo * intensity * diffuse;
		}
#else
		color = vec3(u, 0, v);
#endif
	}

	return color/float(pc.samples_per_light);
}

vec3 light_contr(Hit hit, Ray ray)
{
	vec3 contr = vec3(0.0);

	// Iterate over all lights
	for (int i = 0; i < pc.lights; i++) {
		// Unique (but same) seed for each light
		ray_seed = fract(sin(i * 4325874.3423) * 4398.4324);
		// ray_seed = 0.0;

		// TODO: samlpe method for each light
		uint light_index = floatBitsToUint(light_indices.data[i]);

		float a = lights.data[light_index + 1].x;
		float b = lights.data[light_index + 1].y;
		float c = lights.data[light_index + 1].z;

		uint ia = floatBitsToUint(a);
		uint ib = floatBitsToUint(b);
		uint ic = floatBitsToUint(c);

		vec3 v1 = vertices.data[2 * ia].xyz;
		vec3 v2 = vertices.data[2 * ib].xyz;
		vec3 v3 = vertices.data[2 * ic].xyz;
	
		contr += single_area_light_contr(hit, ray, v1, v2, v3);
	}

	return contr/float(pc.lights);
}

vec3 single_color_at(Ray ray)
{
	// Get closest object
	Hit hit = closest_object(ray);
	if (hit.object != -1)
		return light_contr(hit, ray);

	return hit.mat.albedo;
}

vec3 color_at(Ray ray)
{
	// Get closest object
	Hit hit = closest_object(ray);
	if (hit.object != -1 && hit.mat.shading != SHADING_TYPE_EMMISIVE) {
		vec3 color = light_contr(hit, ray);
		return color;
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
	
	// Get index
	uint index = y0 * viewport.width + x0;

	// Set seed
	float rx = fract(sin(x0 * 1234.56789 + y0) * PHI);
	float ry = fract(sin(y0 * 9876.54321 + x0));
	ray_seed = rx + ry;
	
	vec3 color = vec3(0.0);

#if 1

	for (int i = 0; i < pc.samples_per_pixel; i++) {
		// Random offset
		float x = gold_noise(vec2(x0, y0), ray_seed);
		ray_seed = fract(ray_seed - 10 * sin(ray_seed));
		float y = gold_noise(vec2(x0, y0), ray_seed);
		ray_seed = fract(ray_seed - 10 * sin(ray_seed));

		vec2 offset = vec2(0.5) - vec2(x, y);

		// Create the ray
		vec2 pixel = vec2(x0, y0) + vec2(0.5, 0.5);
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

		// Light transport
		color += color_at(ray);
	}
	
	color /= float(pc.samples_per_pixel);

#else

	rx = fract(sin(x0 * 1234.56789 + y0) * PHI);
	ry = fract(sin(y0 * 9876.54321 + x0));
	color = vec3(rx, 0, ry);

#endif

	float gamma = 2.2;
	color = pow(color, vec3(1/gamma));

	frame.pixels[index] = cast_color(color);
}
