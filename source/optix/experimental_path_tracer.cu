#include "../../source/amadeus/experimental_path_tracer.cuh"

// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/cuda/brdf.cuh"
#include "../../include/cuda/material.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/matrix.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/optix/sbt.cuh"

using namespace kobra::cuda;
using namespace kobra::optix;

extern "C"
{
	__constant__ ExperimentalPathTracerParameters parameters;
}

// TODO: launch parameter for ray depth

// Local constants
static const float eps = 1e-3f;

// Interpolate triangle values
template <class T>
KCUDA_INLINE KCUDA_DEVICE
T interpolate(const T &a, const T &b, const T &c, float2 bary)
{
	return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

// Compute hit point
static KCUDA_INLINE KCUDA_DEVICE
float3 calculate_intersection(Hit *hit, glm::uvec3 triangle, float2 bary)
{
	glm::vec3 a = hit->vertices[triangle.x].position;
	glm::vec3 b = hit->vertices[triangle.y].position;
	glm::vec3 c = hit->vertices[triangle.z].position;
	glm::vec3 x = interpolate(a, b, c, bary);
	x = hit->model * glm::vec4(x, 1.0f);
	return { x.x, x.y, x.z };
}

// Calculate hit normal
static KCUDA_DEVICE
float3 calculate_normal
		(Hit *hit_data, const _material &mat, glm::uvec3 triangle,
		float2 bary, glm::vec2 uv, bool &entering)
{
	glm::vec3 a = hit_data->vertices[triangle.x].position;
	glm::vec3 b = hit_data->vertices[triangle.y].position;
	glm::vec3 c = hit_data->vertices[triangle.z].position;

	// TODO: compute cross, then transform?
	glm::vec3 e1 = b - a;
	glm::vec3 e2 = c - a;

	e1 = hit_data->model * glm::vec4(e1, 0.0f);
	e2 = hit_data->model * glm::vec4(e2, 0.0f);

	glm::vec3 gnormal = glm::normalize(glm::cross(e1, e2));

	float3 ng = { gnormal.x, gnormal.y, gnormal.z };
	if (dot(ng, optixGetWorldRayDirection()) > 0.0f) {
		ng = -ng;
		entering = false;
	} else {
		entering = true;
	}

	ng = normalize(ng);

	a = hit_data->vertices[triangle.x].normal;
	b = hit_data->vertices[triangle.y].normal;
	c = hit_data->vertices[triangle.z].normal;

	gnormal = interpolate(a, b, c, bary);
	gnormal = hit_data->model * glm::vec4(gnormal, 0.0f);

	float3 normal = { gnormal.x, gnormal.y, gnormal.z };
	if (dot(normal, ng) < 0.0f)
		normal = -normal;

	normal = normalize(normal);

	if (mat.textures.has_normal) {
		float4 n4 = tex2D <float4> (mat.textures.normal, uv.x, uv.y);
		float3 n = 2 * make_float3(n4.x, n4.y, n4.z) - 1;

		// Tangent and bitangent
		a = hit_data->vertices[triangle.x].tangent;
		b = hit_data->vertices[triangle.y].tangent;
		c = hit_data->vertices[triangle.z].tangent;

		glm::vec3 gtangent = interpolate(a, b, c, bary);
		gtangent = hit_data->model * glm::vec4(gtangent, 0.0f);

		a = hit_data->vertices[triangle.x].bitangent;
		b = hit_data->vertices[triangle.y].bitangent;
		c = hit_data->vertices[triangle.z].bitangent;

		glm::vec3 gbitangent = interpolate(a, b, c, bary);
		gbitangent = hit_data->model * glm::vec4(gbitangent, 0.0f);

		gtangent = glm::normalize(hit_data->model * glm::vec4(gtangent, 0.0f));
		gbitangent = glm::normalize(hit_data->model * glm::vec4(gbitangent, 0.0f));

		float3 tangent = { gtangent.x, gtangent.y, gtangent.z };
		float3 bitangent = { gbitangent.x, gbitangent.y, gbitangent.z };

		// TODO: get rid of this
		mat3 tbn = mat3(
			normalize(tangent),
			normalize(bitangent),
			normalize(normal)
		);

		normal = normalize(tbn * n);
	}

	return normal;
}

// Calculate relevant material data for a hit
static KCUDA_INLINE KCUDA_DEVICE
Material calculate_material(const _material &mat, glm::vec2 uv)
{
	Material material;
	material.diffuse = mat.diffuse;
	material.specular = mat.specular;
	material.emission = mat.emission;
	material.ambient = mat.ambient; // TODO: remove
	material.shininess = mat.shininess; // TODO: remove
	material.roughness = mat.roughness;
	material.refraction = mat.refraction;
	material.type = mat.type;

	if (mat.textures.has_diffuse) {
		float4 d4 = tex2D <float4> (mat.textures.diffuse, uv.x, uv.y);
		material.diffuse = make_float3(d4);
	}

	if (mat.textures.has_specular) {
		float4 s4 = tex2D <float4> (mat.textures.specular, uv.x, uv.y);
		material.specular = make_float3(s4);
	}

	if (mat.textures.has_emission) {
		float4 e4 = tex2D <float4> (mat.textures.emission, uv.x, uv.y);
		material.emission = make_float3(e4);
	}

	if (mat.textures.has_roughness) {
		float4 r4 = tex2D <float4> (mat.textures.roughness, uv.x, uv.y);
		material.roughness = r4.x;
	}

	return material;
}

// Check light visibility
struct LightVisibility {
	float distance;
	int32_t index;
};

KCUDA_INLINE KCUDA_DEVICE
LightVisibility  query_occlusion(float3 origin, float3 dir, float R, bool no_hit)
{
	static float eps = 0.05f;

	LightVisibility lv { -1.0f, -1 };
	unsigned int j0, j1;
	pack_pointer(&lv, j0, j1);

	int flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
	if (no_hit) {
		flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
		flags |= OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
	}

	optixTrace(parameters.traversable,
		origin, dir,
		0, R - eps, 0,
		OptixVisibilityMask(0xFF),
		flags,
		1, 2, 1, j0, j1
	);

	return lv;
}

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 sample_light(TriangleLight light, Seed seed)
{
	float3 rand = pcg3f(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);

	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}

	return light.a + u * light.ab + v * light.ac;
}

KCUDA_INLINE KCUDA_DEVICE
float power_heuristic(float fpdf, float gpdf)
{
	return (fpdf * fpdf) / (fpdf * fpdf + gpdf * gpdf);
}

KCUDA_INLINE KCUDA_DEVICE
float3 Ld_light(const TriangleLight &light, const SurfaceHit &sh, int index, Seed seed)
{
	float3 Ld = make_float3(0.0f);
	Shading shading = eDiffuse;

	float3 Li = light.intensity;
	if (length(Li) <= 0.0f) // TODO: is black method...
		return Ld;

	// Light sampling
	float3 point = sample_light(light, seed);
	float3 wi = normalize(point - sh.x);
	float R = length(point - sh.x);

	LightVisibility visible = query_occlusion(sh.x, wi, 1.5 * R, false);
	if (visible.index == index) {
		float3 f = brdf(sh, wi, eDiffuse);
		float geometric = abs(dot(sh.n, wi));
		float light_dot = abs(dot(light.normal(), wi));

		float pdf_brdf = pdf(sh, wi, shading);
		float pdf_nee = (R * R)/(light.area() * light_dot);
		float mis_nee_weight = power_heuristic(pdf_nee, pdf_brdf);

		Ld += f * geometric * Li * mis_nee_weight/pdf_nee;
	}

	// BSDF sampling
	float pdf_brdf;
	float3 f = eval(sh, wi, pdf_brdf, shading, seed);
	if (length(f) <= 0 || pdf_brdf <= 0)
		return Ld;

	visible = query_occlusion(sh.x, wi, 1e6f, false);
	if (visible.index == index) {
                // TODO: get distance from query...
		R = visible.distance; // light.intersects(sh.x, wi);
		float geometric = abs(dot(sh.n, wi));
		float light_dot = abs(dot(light.normal(), wi));
		float pdf_nee = (R * R)/(light.area() * light_dot);

		float mis_brdf_weight = power_heuristic(pdf_brdf, pdf_nee);
		Ld += f * geometric * Li * mis_brdf_weight/pdf_brdf;
	}

	return Ld;
}

// Get direct lighting for environment map
KCUDA_DEVICE
float3 Ld_Environment(const SurfaceHit &sh, Seed &seed)
{
	float3 Ld = make_float3(0.0f);
	Shading shading = eDiffuse;

	// Light sampling: sample random direction
	seed = rand_uniform_3f(seed);
	float theta = acosf(sqrtf(1.0f - fract(seed.x)));
	float phi = 2.0f * M_PI * fract(seed.y);

	float3 wi = make_float3(
		sinf(theta) * cosf(phi),
		sinf(theta) * sinf(phi),
		cosf(theta)
	);

	float u = atan2(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(wi.y)/M_PI + 0.5f;

	float4 sample = tex2D <float4> (parameters.environment_map, u, v);
	float3 Li = make_float3(sample);

	LightVisibility visible = query_occlusion(sh.x, wi, 1e6f, false);
	if (visible.index == -1) {
		float3 f = brdf(sh, wi, eDiffuse);
		float geometric = abs(dot(sh.n, wi));
		float pdf_brdf = pdf(sh, wi, shading);
		float pdf_nee = 1/(4.0f * M_PI);
		float mis_nee_weight = power_heuristic(pdf_nee, pdf_brdf);

		Ld += f * geometric * Li * mis_nee_weight/pdf_nee;
	}

	// BSDF sampling
	float pdf_brdf;
	float3 f = eval(sh, wi, pdf_brdf, shading, seed);
	if (length(f) <= 0 || pdf_brdf <= 0)
		return Ld;

	visible = query_occlusion(sh.x, wi, 1e6f, false);
	if (visible.index == -1) {
		float u = atan2(wi.x, wi.z)/(2.0f * M_PI) + 0.5f;
		float v = asin(wi.y)/M_PI + 0.5f;

		float4 sample = tex2D <float4> (parameters.environment_map, u, 1 - v);
		float3 Li = make_float3(sample);

		float geometric = abs(dot(sh.n, wi));
		float pdf_nee = 1/(4.0f * M_PI);

		float mis_brdf_weight = power_heuristic(pdf_brdf, pdf_nee);
		Ld += f * geometric * Li * mis_brdf_weight/pdf_brdf;
	}

	return Ld;
}

// Trace ray into scene and get relevant information
static KCUDA_DEVICE
float3 Ld(const SurfaceHit &sh, Seed seed)
{
	int total_count = parameters.lights.tri_count
		+ parameters.has_environment_map;
	int index = rand_uniform(total_count, seed);

	float3 Ld = make_float3(0.0f);
	if (index < parameters.lights.tri_count) {
		TriangleLight light = parameters.lights.tri_lights[index];
		Ld = Ld_light(light, sh, index, seed);
	} else if (parameters.has_environment_map) {
		Ld = Ld_Environment(sh, seed);
	}

	float sampled_pdf = 1.0f/total_count;
	return Ld/sampled_pdf;
}

// Kernel helpers/code blocks
KCUDA_INLINE __device__
void trace(OptixTraversableHandle handle, int hit_program, int stride, float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(handle,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0xFF),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		hit_program, stride, 0,
		i0, i1
	);
}

static KCUDA_INLINE KCUDA_HOST_DEVICE
float2 make_ray(uint3 idx,
		float3 &origin,
		float3 &direction,
		Seed seed)
{
	const float3 U = to_f3(parameters.camera.ax_u);
	const float3 V = to_f3(parameters.camera.ax_v);
	const float3 W = to_f3(parameters.camera.ax_w);

	// TODO: use a Blue noise sequence

	// Jittered halton
	seed = make_float3(idx.x, idx.y, parameters.samples);
	int xoff = rand_uniform(parameters.resolution.x, seed);
	int yoff = rand_uniform(parameters.resolution.y, seed);

	// Compute ray origin and direction
	int index = xoff + yoff * parameters.resolution.x;
	seed.x = parameters.halton_x[index];
	seed.y = parameters.halton_y[index];

	// TODO: mitchell netravali filter
	float2 d = 2.0f * make_float2(
		float(idx.x + seed.x)/parameters.resolution.x,
		float(idx.y + seed.y)/parameters.resolution.y
	) - 1.0f;

	origin = to_f3(parameters.camera.center);
	direction = normalize(d.x * U + d.y * V + W);

	// Return the radius of the sample
	return float2 {seed.x, seed.y};
}

// Accumulatoin helper
template <class T>
static KCUDA_INLINE KCUDA_DEVICE
void accumulate(T &dst, T sample)
{
	if (parameters.accumulate) {
		T prev = dst;
		float count = parameters.samples;
		dst = (prev * count + sample)/(count + 1);
	} else {
		dst = sample;
	}
}

// Mitchell netravali filter
static KCUDA_INLINE KCUDA_DEVICE
float mitchell(float x)
{
	constexpr float B = 1.0f/3.0f;
	constexpr float C = 1.0f/3.0f;

	// x is in the range [-0.5, 0.5]
	if (!(x >= -0.5f && x <= 0.5f)) {
		printf("x = %f\n", x);
	}

	x = abs(4 * x); // Get to the range [0, 2]

	float x3 = x * x * x;
	float x2 = x * x;

	if (x > 1)
		return ((-B - 6*C)*x3 + (6*B + 30*C)*x2 + (-12*B - 48*C) * x + (8*B + 24*C))/6;

	return ((12 - 9*B - 6*C)*x3 + (-18 + 12*B + 6*C)*x2 + (6 - 2*B))/6;
}

// Ray generation kernel
extern "C" __global__ void __raygen__()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * parameters.resolution.x;

	SurfaceHit sh;

	// Trace ray and generate contribution
	unsigned int i0, i1;
	pack_pointer(&sh, i0, i1);

	float3 origin;
	float3 direction;
	float3 seed;

	float2 offset = make_ray(idx, origin, direction, seed);

	// Path trace
	float3 radiance {0.0f, 0.0f, 0.0f};
	float3 beta {1.0f, 1.0f, 1.0f};
	float3 wi = direction;
	float3 x = origin;

	// AOVs for denoising
	glm::vec4 normal;
	glm::vec4 albedo;
	glm::vec4 position;

	int depth = 0;
	for(;; depth++) {
		if (parameters.russian_roulette) {
			if (depth > 20)
				break;
		} else if (depth > parameters.max_depth) {
			break;
		}

		// Trace the ray
		trace(parameters.traversable, 0, 2, x, wi, i0, i1);

		// First any emission
		bool escaped = (length(sh.n) < 1e-6f);
		if (depth == 0 || escaped) {
			radiance += beta * sh.mat.emission;

			// Check if we hit the environment map
			if (escaped)
				break;
		}

		if (depth == 0) {
			normal = {sh.n.x, sh.n.y, sh.n.z, 0.0f};
			// TODO: put albedo even if hit envmap
			albedo = {sh.mat.diffuse.x, sh.mat.diffuse.y, sh.mat.diffuse.z, 1.0f};
			position = {sh.x.x, sh.x.y, sh.x.z, 0.0f};
		}

		// Otherwise compute direct lighting
		radiance += beta * Ld(sh, seed);

		// Generate new ray
		Shading out;
		float pdf;

		float3 f = eval(sh, wi, pdf, out, seed);

		// If the pdf is zero, we can stop
		if (pdf == 0.0f)
			break;

		// Update the info and continue
		beta *= f * abs(dot(wi, sh.n))/pdf;
		x = sh.x;

		// Russian roulette
		float q = 1.0f - min(max(0.05f, beta.y), 1.0f);
		float r = fract(seed.x);

		if (depth > 2 && parameters.russian_roulette && r < q)
			break;
		else if (parameters.russian_roulette)
			beta /= 1.0f - q;
	}

	// Check for NaNs
	// TODO: can we avoid this?
	if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
		radiance = {1, 0, 1};

	// Accumulate and store necessary data
	// TODO: use an iterative algorithm to save stack space
	auto &buffers = parameters.buffers;

	glm::vec4 color = {radiance.x, radiance.y, radiance.z, 1.0f};
	// accumulate(buffers.color[index], color);
	accumulate(buffers.normal[index], normal);
	accumulate(buffers.albedo[index], albedo);

	// TODO: accumulate position?
	buffers.position[index] = position;

	// Filtered color
	float weight = mitchell(offset.x) * mitchell(offset.y);

	if (parameters.samples == 0)
		parameters.weights[index] = 0.0f;

	buffers.color[index] *= parameters.weights[index];
	// buffers.color[index] *= parameters.samples;
	buffers.color[index] += weight * color;
	// buffers.color[index] /= parameters.samples + 1;
	parameters.weights[index] += weight;
	buffers.color[index] /= parameters.weights[index];
}

// Closest hit kernel
extern "C" __global__ void __closesthit__()
{
	// Load all necessary data
	SurfaceHit *sh;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	sh = unpack_pointer <SurfaceHit> (i0, i1);

	Hit *hit = reinterpret_cast <Hit *>
		(optixGetSbtDataPointer());

	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	glm::uvec3 triangle = hit->triangles[primitive_index];

	glm::vec2 uv_a = hit->vertices[triangle.x].tex_coords;
	glm::vec2 uv_b = hit->vertices[triangle.y].tex_coords;
	glm::vec2 uv_c = hit->vertices[triangle.z].tex_coords;
	glm::vec2 uv = interpolate(uv_a, uv_b, uv_c, bary);
	uv.y = 1 - uv.y;

	_material mat = parameters.materials[hit->material_index];
	Material material = calculate_material(mat, uv);

	bool entering;
	float3 wo = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit, mat, triangle, bary, uv, entering);
	float3 x = calculate_intersection(hit, triangle, bary);

	// Offset by normal
	// TODO: use more complex shadow bias functions
	// TODO: an easier check for transmissive objects
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;

	sh->mat = material;
	sh->entering = entering;
	sh->n = n;
	sh->wo = wo;
	sh->x = x;
}

// Miss kernels
extern "C" __global__ void __miss__()
{
	SurfaceHit *sh;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	sh = unpack_pointer <SurfaceHit> (i0, i1);

	// Get direction
	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y)/M_PI + 0.5f;

	float4 c = make_float4(0);
	if (parameters.has_environment_map)
		c = tex2D <float4> (parameters.environment_map, u, 1 - v);

	sh->mat.emission = make_float3(c);
	sh->n = make_float3(0.0f);
}

// Closest hit miss kernel
extern "C" __global__ void __closesthit__shadow()
{
	LightVisibility *lv;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	lv = unpack_pointer <LightVisibility> (i0, i1);

	Hit *hit = reinterpret_cast <Hit *>
		(optixGetSbtDataPointer());

	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	glm::uvec3 triangle = hit->triangles[primitive_index];
	float3 x = calculate_intersection(hit, triangle, bary);

	lv->distance = length(x - optixGetWorldRayOrigin());

	// TODO: enum the miss modes...
	lv->index = -2;
	if (hit->light_index >= 0)
		lv->index = hit->light_index + primitive_index;
}

extern "C" __global__ void __miss__shadow()
{
	LightVisibility *lv;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	lv = unpack_pointer <LightVisibility> (i0, i1);
	lv->index = -1;
}
