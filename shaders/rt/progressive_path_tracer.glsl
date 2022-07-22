#version 450

// Import bindings
#include "bindings.h"

layout (set = 0, binding = MESH_BINDING_PIXELS, std430) buffer Pixels
{
	uint pixels[];
} frame;

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

// Area lights
struct AreaLight {
	vec3 a;
	vec3 ab;
	vec3 ac; // d = a + ab + ac
	vec3 color;
	float power;
};

layout (set = 0, binding = MESH_BINDING_AREA_LIGHTS, std430) buffer AreaLights
{
	int count;

	AreaLight data[];
} area_lights;

/* Lights
layout (set = 0, binding = MESH_BINDING_LIGHTS, std430) buffer Lights
{
	vec4 data[];
} lights;

// Light indices
layout (set = 0, binding = MESH_BINDING_LIGHT_INDICES, std430) buffer LightIndices
{
	uint data[];
} light_indices; */

// Textures
layout (set = 0, binding = MESH_BINDING_ALBEDOS)
uniform sampler2D s2_albedo[MAX_TEXTURES];

layout (set = 0, binding = MESH_BINDING_NORMAL_MAPS)
uniform sampler2D s2_normals[MAX_TEXTURES];

layout (set = 0, binding = MESH_BINDING_ENVIRONMENT)
uniform sampler2D s2_environment;

// Push constants
layout (push_constant) uniform PushConstants
{
	// Viewport
	uint	width;
	uint	height;

	uint	skip;
	uint	xoffset;
	uint	yoffset;

	// Size variables
	uint	triangles;
	uint	lights;

	// Sample counts
	// TODO: make floats
	uint	samples_per_pixel;
	uint	samples_per_surface;
	uint	samples_per_light;

	// Other options
	uint	accumulate;	// TODO: replace with just present
	uint	present;
	uint	total;

	// Other variables
	float	time;

	// Camera
	vec3	camera_position;
	vec3	camera_forward;
	vec3	camera_up;
	vec3	camera_right;

	// scale, aspect
	vec4	properties;
} pc;

// Sample environment map
vec3 sample_environment(vec3 dir)
{
	vec3 white = vec3(1.0);
	vec3 blue = vec3(0.5, 0.7, 1.0);
	return 0.5 * mix(white, blue, clamp(pow(dir.y, 0.5), 0.0, 1.0));

	/* Get uv coordinates
	const float PI = 3.14159265358979323846;

	vec2 uv = vec2(0.0);
	uv.x = atan(dir.x, dir.z) / (2.0 * PI) + 0.5;
	uv.y = asin(dir.y) / PI + 0.5;

	// Get the color
	vec3 tex = texture(s2_environment, uv).rgb;

	return tex; */
}

// Import all modules
#include "../../include/types.hpp"
#include "modules/ray.glsl"
#include "modules/bbox.glsl"
#include "modules/color.glsl"
#include "modules/random.glsl"
#include "modules/material.glsl"
#include "modules/primitives.glsl"
#include "modules/intersect.glsl"

vec3 sample_environment(Ray ray)
{
	return sample_environment(ray.direction);
}

// Sample environment map wih blur
vec3 sample_environment_blur(Ray ray)
{
	int samples = 16;

	vec3 color = vec3(0.0);
	for (int i = 0; i < samples; i++) {
		vec2 uv = vec2(0.0);
		uv.x = atan(ray.direction.x, ray.direction.z) / (2.0 * PI) + 0.5;
		uv.y = asin(ray.direction.y) / PI + 0.5;

		vec2 j = jitter2d(samples, i);
		vec3 tex = texture(s2_environment, uv + 0.025 * j).rgb;
		color += tex;
	}

	return color / float(samples);
}

// Ray-area light intersection
float _intersect_t(AreaLight light, Ray ray)
{
	// Create the two triangles, then intersect
	vec3 v1 = light.a;
	vec3 v2 = light.a + light.ab;
	vec3 v3 = light.a + light.ac;
	vec3 v4 = light.a + light.ab + light.ac;

	Triangle ta = Triangle(v1, v2, v3);
	Triangle tb = Triangle(v2, v3, v4);

	float t1 = _intersect_t(ta, ray);
	float t2 = _intersect_t(tb, ray);

	if (t1 < 0.0 && t2 < 0.0)
		return -1.0;
	if (t1 < 0.0)
		return t2;
	if (t2 < 0.0)
		return t1;

	return min(t1, t2);
}

Intersection intersection_light(Ray r, AreaLight light)
{
	float time = _intersect_t(light, r);

	Intersection it = Intersection(-1.0, vec3(0.0), mat_default());
	if (time < 0.0)
		return it;

	it.time = time;
	it.mat.albedo = light.color;
	it.mat.shading = SHADING_EMISSIVE;
	return it;
}

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

int id(int node)
{
	vec4 prop = bvh.data[node];
	return floatBitsToInt(prop.x);
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
	int	id;

	float	time;
	vec3	point;
	vec3	normal;

	Material mat;
};

// Get closest object
Hit trace(Ray ray)
{
	int min_index = -1;
	int min_id = -1;

	// Starting intersection
	Intersection mini = Intersection(
		1.0/0.0, vec3(0.0),
		mat_default()
	);

	// Traverse BVH as a threaded binary tree
	int node = 0;
	while (node != -1) {
		if (object(node) != -1) {
			// Get object index
			int index = object(node);

			// Get object
			Intersection it = ray_intersect(ray, index);

			// If intersection is valid, update minimum
			if (it.time > 0.0 && it.time < mini.time) {
				min_index = index;
				min_id = id(node);
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

			if ((t > 0.0 && t < mini.time) || inside) {
				// Traverse left child
				node = hit(node);
			} else {
				// Traverse right child
				node = miss(node);
			}
		}
	}

	// Check area lights
	for (int i = 0; i < area_lights.count; i++) {
		AreaLight light = area_lights.data[i];
		Intersection it = intersection_light(ray, light);
		if (it.time > 0.0 && it.time < mini.time) {
			min_index = i;
			min_id = i;
			mini = it;
		}
	}

	// Color of closest object
	vec3 color = vec3(0.0);	// TODO: sample from either texture or gradient
	if (min_index < 0)
		mini.mat.albedo = sample_environment(ray);

	vec3 point = ray.origin + ray.direction * mini.time;

	return Hit(
		min_index,
		min_id,
		mini.time,
		point,
		mini.normal,
		mini.mat
	);
}

// Maximum ray depth
#define MAX_DEPTH 10

// Power heuristic
float power_heuristic(float nf, float fpdf, float ng, float gpdf)
{
	float f = nf * fpdf;
	float g = ng * gpdf;

	return (f * f) / (f * f + g * g);
}

// Fresnel reflectance
float fresnel_dielectric(float cosi, float etai, float etat)
{
	// Swap if necessary
	cosi = clamp(cosi, -1.0, 1.0);
	if (cosi < 0.0) {
		cosi = -cosi;

		float tmp = etai;
		etai = etat;
		etat = tmp;
	}

	float sini = sqrt(max(0.0, 1.0 - cosi * cosi));

	float sint = etai / etat * sini;
	if (sint >= 1.0)
		return 1.0;

	float cost = sqrt(max(0.0, 1.0 - sint * sint));

	float r_parl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	float r_perp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));

	return (r_parl * r_parl + r_perp * r_perp) / 2.0;
}

// Invidual BSDF types
#define BSDF(ftn) void bsdf_##ftn(in Hit hit,	\
		inout Ray ray,			\
		inout float pdf,		\
		inout vec3 beta,		\
		inout float ior)

BSDF(specular_reflection)
{
	float cosi = dot(ray.direction, hit.normal);
	float Fr = fresnel_dielectric(cosi, hit.mat.ior, ior);

	ray.direction = reflect(ray.direction, hit.normal);
	ray.origin = hit.point + hit.normal * 0.001;

	float cos_theta = abs(dot(ray.direction, hit.normal));

	pdf = 1;

	// TODO: fresnel conductors: if ior = 1, then
	// Fr is always equal to 0
	beta *= hit.mat.albedo/abs(cos_theta);
}

BSDF(specular_transmission)
{
	float cosi = dot(ray.direction, hit.normal);
	float Fr = fresnel_dielectric(cosi, hit.mat.ior, ior);

	float eta = ior/hit.mat.ior;
	ray.direction = refract(ray.direction, hit.normal, eta);
	ray.origin = hit.point - hit.normal * 0.001;

	float cos_theta = abs(dot(ray.direction, hit.normal));

	pdf = 1;
	ior = hit.mat.ior;
	beta *= (1 - Fr) * hit.mat.albedo / cos_theta;
}

BSDF(lambertian)
{
	// Assume diffuse
	ray.direction = cosine_weighted_hemisphere(hit.normal);
	ray.origin = hit.point + hit.normal * 0.001;

	pdf = INV_PI * dot(ray.direction, hit.normal);
	beta *= hit.mat.albedo * INV_PI;
}

// Sample a ray from BSDF
void sample_bsdf(in Hit hit, inout Ray ray,
		inout float pdf,
		inout vec3 beta,
		inout float ior)
{
	int shading = hit.mat.shading;
	if (is_type(shading, SHADING_REFLECTION | SHADING_TRANSMISSION)) {
		// Choose reflection or transmission randomly
		//	based on Fresnel reflectance
		float cosi = dot(ray.direction, hit.normal);
		float Fr = fresnel_dielectric(cosi, hit.mat.ior, ior);

		float rand = random();
		if (rand < Fr)
			bsdf_specular_reflection(hit, ray, pdf, beta, ior);
		else
			bsdf_specular_transmission(hit, ray, pdf, beta, ior);
	} else if (is_type(shading, SHADING_REFLECTION)) {
		bsdf_specular_reflection(hit, ray, pdf, beta, ior);
	} else if (is_type(shading, SHADING_TRANSMISSION)) {
		bsdf_specular_transmission(hit, ray, pdf, beta, ior);
	} else if (is_type(shading, SHADING_DIFFUSE)) {
		// TOOD: also microfacet diffuse is a sigma != 0
		bsdf_lambertian(hit, ray, pdf, beta, ior);
	} else {
		pdf = 0.0;
	}

	beta *= abs(dot(hit.normal, ray.direction)) / pdf;
}

// Get pdf of a direction from BSDF
float pdf_bsdf(in Hit hit, in Ray ray, vec3 wi)
{
	int shading = hit.mat.shading;
	if (is_type(shading, SHADING_DIFFUSE)) {
		// Assume diffuse
		// TODO: check same hemisphere
		return INV_PI * dot(wi, hit.normal);
	}

	return 0.0;
}

// Direct illumination
vec3 direct_illumination(Hit hit, Ray ray)
{
	// Direct light contribution
	vec3 direct_contr = vec3(0.0);
	vec3 beta = vec3(0.0);

	// Direct illumination
	uvec3 seed = floatBitsToUint(vec3(pc.time, hit.point.x, hit.point.y));
	uint i = randuint(seed, area_lights.count);

	// TODO: some way to check ray-light intersection (can be independent of
	// trace :))
	// uint light_index = light_indices.data[i];
	// int light_object = int(floatBitsToUint(lights.data[light_index + 1].w));

	AreaLight light = area_lights.data[i];

	// Ray to use
	Ray r = ray;

	// Random 2D point on light
	vec3 rand = random_sphere();
	float u = fract(rand.x);
	float v = fract(rand.y);

	vec3 lpos = light.a + u * light.ab + v * light.ac;

	// Try to connect to the light
	vec3 pos = hit.point + hit.normal * 0.001;
	vec3 dir = normalize(lpos - pos);

	float pdf = pdf_bsdf(hit, r, dir);
	if (pdf == 0.0)
		return vec3(0.0);

	// TODO: remove the extra arguments
	Ray shadow_ray = Ray(pos, dir, 1.0, 1.0);

	Hit shadow_hit = trace(shadow_ray);

	// if (shadow_hit.id == light_object) {
	if (distance(shadow_hit.point, lpos) < 0.001) {
		// Light contribution directly
		float d = distance(lpos, hit.point);
		float cos_theta = max(0.0, dot(hit.normal, dir));

		// TODO: get the pdf from anotehr function
		direct_contr += light.color * light.power * hit.mat.albedo
			* cos_theta * (1/d)
			* power_heuristic(float(pc.samples_per_light), 0.25 * INV_PI, 1, pdf)
			* (4 * PI);
	}

	return direct_contr;
}

// Total color for a ray
vec3 color_at(Ray ray)
{
	vec3 contribution = vec3(0.0);
	vec3 beta = vec3(1.0);

	// Index of refraction
	float ior = 1.0;

	Ray r = ray;
	for (int i = 0; i < MAX_DEPTH; i++) {
		// Find closest object
		// TODO: refactor to trace
		Hit hit = trace(r);

		// Special case intersection
		// TODO: deal with in the direct lighting function
		// TODO: shading emissive --> shading light
		if (hit.object == -1 || hit.mat.shading == SHADING_EMISSIVE) {
			contribution += beta * hit.mat.albedo;
			break;
		}

		// Direct illumination
		vec3 direct_contr = direct_illumination(hit, r);
		contribution += beta * direct_contr;

		// Sample BSDF
		float pdf = 0.0;
		sample_bsdf(hit, r, pdf, beta, ior);

		if (pdf == 0.0)
			break;

		// Russian roulette
		if (i > 2) {
			float q = max(1.0 - beta.y, 0.05);
			if (random() < q)
				break;
			beta /= (1.0 - q);
		}
	}

	return clamp(contribution, 0.0, 1.0);
}

void main()
{
	// Offset from space origin
	uint y0 = pc.skip * gl_WorkGroupID.y + pc.xoffset;
	uint x0 = pc.skip * gl_WorkGroupID.x + pc.yoffset;

	// Return if out of bounds
	if (y0 >= pc.height || x0 >= pc.width)
		return;

	// Get index
	uint index = y0 * pc.width + x0;

	// Set seed
	float rx = fract(sin(x0 * 12 + y0) * PHI);
	float ry = fract(sin(y0 * 98 + x0));

	// Initialiize the random seed
	random_seed = vec3(rx, ry, fract((rx + ry)/pc.time));
	vec2 dimensions = vec2(pc.width, pc.height);

	// Create the ray
	vec2 pixel = vec2(x0, y0) + random_sphere().xy/2.0;
	vec2 uv = pixel / dimensions;

	Ray ray = make_ray(uv,
		pc.camera_position,
		pc.camera_forward,
		pc.camera_up,
		pc.camera_right,
		pc.properties.x,
		pc.properties.y
	);

	// Progressive rendering
	vec3 color = color_at(ray);
	vec3 pcolor = cast_color(frame.pixels[index]);
	pcolor = pow(pcolor, vec3(2.2));
	float t = 1.0f/(1.0f + pc.present);
	color = mix(pcolor, color, t);
	color = pow(color, vec3(1.0/2.2));
	frame.pixels[index] = cast_color(color);
}
