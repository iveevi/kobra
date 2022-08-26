// Standard headers
#include <cstdint>

// #define KCUDA_DEBUG

// Engine headers
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/cuda/random.cuh"
#include "../../include/cuda/brdf.cuh"
#include "../../include/layers/optix_tracer_common.cuh"

using kobra::optix_rt::HitGroupData;
using kobra::optix_rt::MissData;
using kobra::optix_rt::AreaLight;

using kobra::cuda::Material;
using kobra::cuda::GGX;

extern "C"
{
	__constant__ kobra::optix_rt::Params params;
}

// Helper functionss
template <class T>
static __forceinline__ __device__ T *unpack_point(uint32_t i0, uint32_t i1)
{
	const uint64_t uptr = static_cast <uint64_t> (i0) << 32 | i1;
	T *ptr = reinterpret_cast <T *> (uptr);
	return ptr;
}

template <class T>
static __forceinline__ __device__ void pack_pointer(T * ptr, uint32_t &i0, uint32_t &i1)
{
	const uint64_t uptr = reinterpret_cast <uint64_t> (ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void make_ray
		(uint3 idx, uint3 dim,
		 float3 &origin, float3 &direction,
		 float3 &seed)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;

	int index = idx.x + params.image_width * idx.y;

	// TODO: jittered halton sequence
	float xoffset = fract(random3(seed).x) - 0.5f;
	float yoffset = fract(random3(seed).y) - 0.5f;

	float2 d = 2.0f * make_float2(
		float(idx.x + xoffset)/dim.x,
		float(idx.y + yoffset)/dim.y
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

// Ray packet data
struct RayPacket {
	float3 throughput;
	float3 value;
	float3 seed;
	float ior;
	int depth;
};

extern "C" __global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// Pack payload
	RayPacket ray_packet;
	ray_packet.throughput = {1.0f, 1.0f, 1.0f};
	ray_packet.value = {0.0f, 0.0f, 0.0f};
	ray_packet.seed = {float(idx.x), float(idx.y), params.time};
	ray_packet.ior = 1.0f;
	ray_packet.depth = 0;

	// Generate ray
	float3 ray_origin;
	float3 ray_direction;

	make_ray(idx, dim, ray_origin, ray_direction, ray_packet.seed);

	unsigned int i0, i1;
	pack_pointer(&ray_packet, i0, i1);
	
	// Launch
	optixTrace(params.handle,
		ray_origin, ray_direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 0, 0,
		i0, i1
	);

	// Check value
	float3 pixel = ray_packet.value;
	assert(!isnan(pixel.x) && !isnan(pixel.y) && !isnan(pixel.z));
	if (isnan(pixel.x) || isnan(pixel.y) || isnan(pixel.z))
		pixel = {0.0f, 0.0f, 0.0f};

	// Record the results
	int index = idx.x + params.image_width * idx.y;
	// params.pbuffer[index] = ray_packet.value;
	params.pbuffer[index] = (pixel + params.pbuffer[index] * params.accumulated)/(params.accumulated + 1);
	params.image[index] = kobra::cuda::make_color(params.pbuffer[index]);
}

extern "C" __global__ void __miss__radiance()
{
	// Background color based on ray direction
	// TODO: implement background
	MissData *miss_data = reinterpret_cast <MissData *> (optixGetSbtDataPointer());

	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z) / (2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y) / M_PI + 0.5f;

	float4 c = tex2D <float4> (miss_data->bg_tex, u, v);

	// Transfer to payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_point <RayPacket> (i0, i1);
	rp->value += rp->throughput * make_float3(c);
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_point <bool> (i0, i1);
	*vis = true;
}

struct mat3 {
	// Column major
	float m[9];

	__device__ __forceinline__ mat3() {}

	__device__ __forceinline__ mat3(float3 c1, float3 c2, float3 c3) {
		// Store in column major order
		m[0] = c1.x; m[3] = c2.x; m[6] = c3.x;
		m[1] = c1.y; m[4] = c2.y; m[7] = c3.y;
		m[2] = c1.z; m[5] = c2.z; m[8] = c3.z;
	}
};

__device__ __forceinline__ float3 operator*(mat3 m, float3 v)
{
	return make_float3(
		m.m[0] * v.x + m.m[3] * v.y + m.m[6] * v.z,
		m.m[1] * v.x + m.m[4] * v.y + m.m[7] * v.z,
		m.m[2] * v.x + m.m[5] * v.y + m.m[8] * v.z
	);
}

#define MAX_DEPTH 5

// Evaluate BRDF of material
__device__ float3 brdf(const Material &mat, float3 n, float3 wi,
		float3 wo, float ior, Shading out)
{
	float3 o = GGX::brdf(mat, n, wi, wo, ior, out) + mat.diffuse/M_PI;
	/* if (out & Shading::eTransmission) assert(!isnan(o.x) && !isnan(o.y) && !isnan(o.z));
	else assert(!isnan(o.x) && !isnan(o.y) && !isnan(o.z)); */
	return o;
}

// Evaluate PDF of BRDF
__device__ float pdf(const Material &mat, float3 n, float3 wi,
		float3 wo, Shading out)
{
	float o = GGX::pdf(mat, n, wi, wo, out);
	/* if (out & Shading::eTransmission) assert(!isnan(o));
	else assert(!isnan(o)); */
	return o;
}

// Sample BRDF
__device__ float3 sample(const Material &mat, float3 n, float3 wo,
		float ior, float3 &seed, Shading &out)
{
	float3 o = GGX::sample(mat, n, wo, ior, seed, out);
	/* if (out & Shading::eTransmission) assert(!isnan(o.x) && !isnan(o.y) && !isnan(o.z));
	else assert(!isnan(o.x) && !isnan(o.y) && !isnan(o.z)); */
	return o;
}

// Power heurestic
static const float p = 2.0f;

__device__ float power(float pdf_f, float pdf_g)
{
	float f = pow(pdf_f, p);
	float g = pow(pdf_g, p);

	return f/(f + g);
}

// Area light methods
__device__ float3 sample_area_light(AreaLight light, float3 &seed)
{
	float3 rand = random3(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);
	return light.a + u * light.ab + v * light.ac;
}

// Check shadow visibility
__device__ bool shadow_visibility(float3 origin, float3 dir, float R)
{
	bool vis = false;
	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	// TODO: max time show be distance to light
	optixTrace(params.handle_shadow,
		origin, dir,
		0, R - 0.01f, 0,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
			| OPTIX_RAY_FLAG_DISABLE_ANYHIT
			| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		params.instances, 0, 1,
		j0, j1
	);

	return vis;
}

// Trace ray into scene and get relevant information
__device__ float3 Ld(HitGroupData *hit_data, float3 x, float3 wo, float3 n,
		Material mat, float ior, float3 &seed)
{
	if (hit_data->n_area_lights == 0)
		return float3 {0.0f, 0.0f, 0.0f};

	float3 contr_nee {0.0f};
	float3 contr_brdf {0.0f};

	// Random area light for NEE
	random3(seed);
	unsigned int i = seed.x * hit_data->n_area_lights;
	i = i % hit_data->n_area_lights;
	AreaLight light = hit_data->area_lights[i];

	// NEE
	float3 lpos = sample_area_light(light, seed);
	float3 wi = normalize(lpos - x);
	float R = length(lpos - x);

	print("===NEE brdf===\n");
	float3 f = brdf(mat, n, wi, wo, ior, mat.type) * max(dot(n, wi), 0.0f);

	float ldot = abs(dot(light.normal(), wi));
	if (ldot > 1e-6) {
		float pdf_light = (R * R)/(light.area() * ldot);

		// TODO: how to decide ray type for this?
		float pdf_brdf = pdf(mat, n, wi, wo, mat.type);

		bool vis = shadow_visibility(x, wi, R);
		if (pdf_light > 1e-9 && vis) {
			float weight = power(pdf_light, pdf_brdf);
			float3 intensity = light.intensity;
			contr_nee += weight * f * intensity/pdf_light;
		}
	}

	// BRDF
	Shading out;
	wi = sample(mat, n, wo, 1, seed, out);
	if (dot(wi, n) <= 0.0f)
		return contr_nee;
	
	print("===BRDF brdf===\n");
	f = brdf(mat, n, wi, wo, ior, out) * max(dot(n, wi), 0.0f);

	float pdf_brdf = pdf(mat, n, wi, wo, out);
	float pdf_light = 0.0f;

	// TODO: need to check intersection for lights specifically (and
	// arbitrary ones too?)
	float ltime = light.intersects(x, wi);
	if (ltime <= 0.0f)
		return contr_nee;

	R = ltime;
	pdf_light = (R * R)/(light.area() * abs(dot(light.normal(), wi)));
	if (pdf_light > 1e-9 && pdf_brdf > 1e-9) {
		float weight = power(pdf_brdf, pdf_light);
		float3 intensity = light.intensity;
		contr_brdf += weight * f * intensity/pdf_brdf;
	}

	return contr_nee + contr_brdf;
}

// Interpolate triangle values
template <class T>
__device__ T interpolate(T *arr, uint3 triagle, float2 bary)
{
	T a = arr[triagle.x];
	T b = arr[triagle.y];
	T c = arr[triagle.z];

	return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

// Sample from a texture
static __forceinline__ __device__ float4 sample_texture
		(HitGroupData *hit_data, cudaTextureObject_t tex, uint3 triangle, float2 bary)
{
	float2 uv = interpolate(hit_data->texcoords, triangle, bary);
	return tex2D <float4> (tex, uv.x, 1 - uv.y);
}

// Calculate hit normal
static __forceinline__ __device__ float3 calculate_normal
		(HitGroupData *hit_data, uint3 triangle, float2 bary)
{
	float3 e1 = hit_data->vertices[triangle.y] - hit_data->vertices[triangle.x];
	float3 e2 = hit_data->vertices[triangle.z] - hit_data->vertices[triangle.x];
	float3 ng = cross(e1, e2);

	if (dot(ng, optixGetWorldRayDirection()) > 0.0f)
		ng = -ng;

	ng = normalize(ng);

	float3 normal = interpolate(hit_data->normals, triangle, bary);
	if (dot(normal, ng) < 0.0f)
		normal -= 2.0f * dot(normal, ng) * ng;

	normal = normalize(normal);

	if (hit_data->textures.has_normal) {
		float4 n4 = sample_texture(hit_data,
			hit_data->textures.normal,
			triangle, bary
		);

		float3 n = 2 * make_float3(n4.x, n4.y, n4.z) - 1;

		// Tangent and bitangent
		float3 tangent = interpolate(hit_data->tangents, triangle, bary);
		float3 bitangent = interpolate(hit_data->bitangents, triangle, bary);

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
__device__ void calculate_material
		(HitGroupData *hit_data,
		Material &mat,
		uint3 triangle, float2 bary)
{
	if (hit_data->textures.has_diffuse) {
		mat.diffuse = make_float3(
			sample_texture(hit_data,
				hit_data->textures.diffuse,
				triangle, bary
			)
		);
	}

	if (hit_data->textures.has_roughness) {
		mat.roughness = sample_texture(hit_data,
			hit_data->textures.roughness,
			triangle, bary
		).x;
	}
}

extern "C" __global__ void __closesthit__radiance()
{
	// Get payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_point <RayPacket> (i0, i1);

	if (rp->depth > MAX_DEPTH)
		return;

	// Get data from the SBT
	HitGroupData *hit_data = reinterpret_cast <HitGroupData *> (optixGetSbtDataPointer());

	// TODO: check for light, not just emissive material
	if (hit_data->material.type == Shading::eEmissive) {
		rp->value += rp->throughput * hit_data->material.emission;
		rp->throughput = {0, 0, 0};
		return;
	}

	// Calculate relevant data for the hit
	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	uint3 triangle = hit_data->triangles[primitive_index];

	Material material = hit_data->material;
	calculate_material(hit_data, material, triangle, bary);

	float3 wo = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit_data, triangle, bary);
	float3 x = interpolate(hit_data->vertices, triangle, bary);

	float3 direct = Ld(hit_data, x + 1e-3f * n, wo, n, material, rp->ior, rp->seed);

	// Transfer to payload
	rp->value += direct * rp->throughput;

	// Generate new ray
	Shading out;
	print("===Recursive brdf===\n");
	float3 wi = sample(material, n, wo, rp->ior, rp->seed, out);
	if (length(wi) < 1e-9)
		return;

	float pdf = ::pdf(material, n, wi, wo, out);

	if (pdf <= 1e-9)
		return;

	float3 f = brdf(material, n, wi, wo, rp->ior, out) * abs(dot(n, wi));
	float3 T = f/pdf;

	// Russian roulette
	float p = max(rp->throughput.x, max(rp->throughput.y, rp->throughput.z));
	float q = 1 - min(1.0f, p);
	if (fract(rp->seed.x) < q)
		return;

	rp->throughput *= T/(1 - q);
	rp->depth++;

	// Recursive raytrace
	float3 offset = 1e-3f * n;
	if (out & Shading::eTransmission)
		offset = -offset;

	// rp->value = GGX::brdf(material, n, wi, wo, rp->ior, out, true);
	// return;

	// Update ior
	rp->ior = material.refraction;

	// Recurse
	optixTrace(params.handle,
		x + offset, wi,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		0, 0, 0,
		i0, i1
	);
}

extern "C" __global__ void __closesthit__shadow() {}
