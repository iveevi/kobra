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
	vec3 intensity;
};

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

// Import all modules
#include "../../include/types.hpp"
#include "modules/ray.glsl"
#include "modules/bbox.glsl"
#include "modules/color.glsl"
#include "modules/random.glsl"
#include "modules/material.glsl"
#include "modules/primitives.glsl"
#include "modules/environment.glsl"
#include "modules/intersect.glsl"
#include "modules/bvh.glsl"

// Maximum ray depth
#define MAX_DEPTH 10

vec3 sample_triangle(vec3 v1, vec3 v2, vec3 v3, float strata, float i)
{
	// Get random vec in [0, 1]
	vec2 r = jitter2d(strata, i);
	float s = sqrt(r.x);

	float u = 1.0 - s;
	float v = r.y * s;

	// Edge vectors
	vec3 e1 = v2 - v1;
	vec3 e2 = v3 - v1;

	vec3 p = v1 + u * e1 + v * e2;

	return p;
}

// Sample random point from an area light
vec3 sample_area_light(uint li, int si)
{
	// Get light position
	uvec4 i = VERTEX_STRIDE * floatBitsToUint(lights.data[li + 1]);

	vec3 v1 = vertices.data[i.x].xyz;
	vec3 v2 = vertices.data[i.y].xyz;
	vec3 v3 = vertices.data[i.z].xyz;

	// Get random point in triangle
	return sample_triangle(
		v1, v2, v3,
		sqrt(pc.samples_per_light), si
	);
}

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
	uint i = randuint(seed, pc.lights);

	uint light_index = light_indices.data[i];
	int light_object = int(floatBitsToUint(lights.data[light_index + 1].w));

	// Ray to use
	Ray r = ray;

	float inv_lsamples = 1.0/pc.samples_per_light;
	for (int j = 0; j < pc.samples_per_light; j++) {
		vec3 light_position = sample_area_light(light_index, j);

		// Try to connect to the light
		vec3 pos = hit.point + hit.normal * 0.001;
		vec3 dir = normalize(light_position - pos);

		float pdf = pdf_bsdf(hit, r, dir);
		if (pdf == 0.0)
			continue;

		// TODO: remove the extra arguments
		Ray shadow_ray = Ray(pos, dir, 1.0, 1.0);

		Hit shadow_hit = trace(shadow_ray);
		// if (shadow_hit.id == light_object) {
		if (distance(shadow_hit.point, light_position) < 0.001) {
			// Light contribution directly
			float d = distance(light_position, hit.point);
			float cos_theta = max(0.0, dot(hit.normal, dir));
			vec3 lcolor = vec3(1.0);

			// TODO: get the pdf from anotehr function
			direct_contr += lcolor * hit.mat.albedo
				* cos_theta * (1/d)
				* power_heuristic(float(pc.samples_per_light), 0.25 * INV_PI, 1, pdf)
				* inv_lsamples * (4 * PI);
		}
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
			contribution += hit.mat.albedo;
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
