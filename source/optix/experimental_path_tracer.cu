#include "../../source/amadeus/experimental_path_tracer.cuh"

// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/cuda/brdf.cuh"
#include "../../include/cuda/material.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/matrix.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/optix/lighting.cuh"
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

// Check shadow visibility
KCUDA_INLINE __device__
bool is_occluded(OptixTraversableHandle handle, float3 origin, float3 dir, float R)
{
	static float eps = 0.05f;

	bool vis = true;

	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	optixTrace(handle,
		origin, dir,
		0, R - eps, 0,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
			| OPTIX_RAY_FLAG_DISABLE_ANYHIT
			| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		0, 0, 1,
		j0, j1
	);

	return vis;
}

// Get direct lighting for environment map
KCUDA_DEVICE
float3 Ld_Environment(const SurfaceHit &sh, float &pdf, Seed seed)
{
	// TODO: sample in UV space instead of direction...
	static const float WORLD_RADIUS = 10000.0f;

	// Sample random direction
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
	pdf = 1.0f/(4.0f * M_PI);

	// NEE
	bool occluded = is_occluded(parameters.traversable, sh.x, wi, WORLD_RADIUS);
	if (occluded)
		return make_float3(0.0f);

	// TODO: method for abs dot in surface hit
	float3 f = brdf(sh, wi, eDiffuse) * abs(dot(sh.n, wi));

	// TODO: how to decide ray type for this?
	return f * Li;
}

// Trace ray into scene and get relevant information
static KCUDA_DEVICE
float3 Ld(const SurfaceHit &sh, Seed seed)
{
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.tri_count;

	// TODO: parameter for if envmap is used
	int total_count = quad_count + tri_count + parameters.has_environment_map;

	// Regular direct lighting
	unsigned int i = rand_uniform(seed) * total_count;

	float3 contr = {0.0f};
	float pdf = 1.0f/total_count;

	// TODO: importance sample power
	float light_pdf = 0.0f;
	if (i < quad_count) {
		QuadLight light = parameters.lights.quad_lights[i];
		contr = Ld_light(parameters.traversable, light, sh, light_pdf, seed);
	} else if (i < quad_count + tri_count) {
		int ni = i - quad_count;
		TriangleLight light = parameters.lights.tri_lights[ni];
		contr = Ld_light(parameters.traversable, light, sh, light_pdf, seed);
	} else {
		// Environment light
		// TODO: imlpement PBRT's better importance sampling
		contr = Ld_Environment(sh, light_pdf, seed);
	}

	pdf *= light_pdf;

	// TODO: MIS with BRDF sampling

	return contr/pdf;
}

// Kernel helpers/code blocks
KCUDA_INLINE __device__
void trace(OptixTraversableHandle handle, int hit_program, int stride, float3 origin, float3 direction, uint i0, uint i1)
{
	optixTrace(handle,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		hit_program, stride, 0,
		i0, i1
	);
}

static KCUDA_INLINE KCUDA_HOST_DEVICE
void make_ray(uint3 idx,
		float3 &origin,
		float3 &direction,
		float3 &seed)
{
	const float3 U = to_f3(parameters.camera.ax_u);
	const float3 V = to_f3(parameters.camera.ax_v);
	const float3 W = to_f3(parameters.camera.ax_w);

	// TODO: use a Blue noise sequence
	// Constant, jittered
	/* Jittered halton
	int xoff = rand(parameters.image_width, seed);
	int yoff = rand(parameters.image_height, seed);

	// Compute ray origin and direction
	float xoffset = parameters.xoffset[xoff];
	float yoffset = parameters.yoffset[yoff];
	radius = sqrt(xoffset * xoffset + yoffset * yoffset)/sqrt(0.5f); */

	// TODO: mitchell netravali filter
	pcg3f(seed);

	float xoffset = (fract(seed.x) - 0.5f);
	float yoffset = (fract(seed.y) - 0.5f);

	float2 d = 2.0f * make_float2(
		float(idx.x + xoffset)/parameters.resolution.x,
		float(idx.y + yoffset)/parameters.resolution.y
	) - 1.0f;

	origin = to_f3(parameters.camera.center);
	direction = normalize(d.x * U + d.y * V + W);
}

// Accumulatoin helper
template <class T>
__forceinline__ __device__
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

	float3 seed = make_float3(
		idx.x + parameters.samples,
		idx.y + parameters.samples,
		parameters.samples
	);

	make_ray(idx, origin, direction, seed);

	// Path trace
	float3 radiance {0.0f, 0.0f, 0.0f};
	float3 beta {1.0f, 1.0f, 1.0f};
	float3 wi = direction;
	float3 x = origin;
	bool specular = false;

	// AOVs for denoising
	glm::vec4 normal;
	glm::vec4 albedo;
	glm::vec4 position;

	int depth = 1;
	for(;; depth++) {
		if (!parameters.russian_roulette
				&& (depth > parameters.max_depth + 1))
			break;

		// Trace the ray
		trace(parameters.traversable, 0, 1, x, wi, i0, i1);

		// Check if we hit the environment map
		if (length(sh.n) < 1e-3f) {
			radiance += beta * sh.mat.emission;
			break;
		}

		if (depth == 1) {
			normal = {sh.n.x, sh.n.y, sh.n.z, 0.0f};
			// TODO: put albedo even if hit envmap
			albedo = {sh.mat.diffuse.x, sh.mat.diffuse.y, sh.mat.diffuse.z, 1.0f};
			position = {sh.x.x, sh.x.y, sh.x.z, 0.0f};
		}

		// Otherwise compute direct lighting
		float3 direct = Ld(sh, seed);
		if (depth == 1 || specular)
			direct += sh.mat.emission;

		// Russian roulette
		float q = 1.0f - min(max(0.05f, beta.y), 1.0f);
		float r = fract(seed.x);

		if (parameters.russian_roulette && r < q) {
			radiance += beta * direct;
			break;
		}

		radiance += beta * direct;
		if (parameters.russian_roulette)
			beta /= 1.0f - q;

		// Generate new ray
		Shading out;
		float pdf;

		float3 f = eval(sh, wi, pdf, out, seed);

		// If the pdf is zero, we can stop
		if (pdf == 0.0f)
			break;

		// Update the info and continue
		beta *= f * abs(dot(wi, sh.n))/pdf;
		specular = (length(sh.mat.specular) > 0);
		x = sh.x;
	}

	// Finally, store the result
	// TODO: avoid this casting: just assume the buffers are float4s...

	// Check for NaNs
	// TODO: can we avoid this?
	if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
		radiance = {1, 0, 1};

	// Accumulate and store necessary data
	// TODO: use an iterative algorithm to save stack space
	auto &buffers = parameters.buffers;

	glm::vec4 color = {radiance.x, radiance.y, radiance.z, 1.0f};
	accumulate(buffers.color[index], color);
	accumulate(buffers.normal[index], normal);
	accumulate(buffers.albedo[index], albedo);

	// TODO: accumulate position?
	buffers.position[index] = position;
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
		c = tex2D <float4> (parameters.environment_map, u, v);

	sh->mat.emission = make_float3(c);
	sh->n = make_float3(0.0f);
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_pointer <bool> (i0, i1);
	*vis = false;
}
