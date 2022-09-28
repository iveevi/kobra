// Standard headers
#include <cstdint>

#define KCUDA_DEBUG

// Engine headers
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/cuda/random.cuh"
#include "../../include/cuda/brdf.cuh"
#include "../../include/layers/optix_tracer_common.cuh"

using kobra::optix_rt::HitGroupData;
using kobra::optix_rt::MissData;
using kobra::optix_rt::QuadLight;
using kobra::optix_rt::TriangleLight;

using kobra::cuda::Material;

using kobra::cuda::GGX;
using kobra::cuda::SpecularTransmission;
using kobra::cuda::SpecularReflection;
using kobra::cuda::FresnelSpecular;

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
		(uint3 idx, uint3 dim, int index,
		 float3 &origin,
		 float3 &direction,
		 float3 &seed,
		 float &radius)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	
	// Jittered halton
	int xoff = rand(params.image_width, seed);
	int yoff = rand(params.image_height, seed);

	// Compute ray origin and direction
	float xoffset = params.xoffset[xoff];
	float yoffset = params.yoffset[yoff];
	radius = sqrt(xoffset * xoffset + yoffset * yoffset)/sqrt(0.5f);

	float2 d = 2.0f * make_float2(
		float(idx.x + xoffset)/dim.x,
		float(idx.y + yoffset)/dim.y
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

// Ray packet data
struct RayPacket {
	float3 diffuse;
	float3 normal;
	float3 throughput;
	float3 value;
	float3 seed;
	float ior;
	int depth;
};

namespace filters {

__forceinline__ __device__
float box(float r)
{
	return 1.0f;
}

__forceinline__ __device__
float triangle(float r)
{
	return max(0.0f, min(1.0f - r, 1.0f));
}

__forceinline__ __device__
float gaussian(float r, float alpha = 2.0f)
{
	return exp(-alpha * r * r) - exp(-alpha);
}

__forceinline__ __device__
float mitchell(float r, float B = 1.0f/3.0f, float C = 1.0f/3.0f)
{
	r = fabs(2.0f * r);
	if (r > 1.0f) {
		return ((-B - 6.0f * C) * r * r * r + (6.0f * B + 30.0f * C) * r * r +
			(-12.0f * B - 48.0f * C) * r + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
	}

	return ((12.0f - 9.0f * B - 6.0f * C) * r * r * r +
		(-18.0f + 12.0f * B + 6.0f * C) * r * r +
		(6.0f - 2.0f * B)) * (1.0f / 6.0f);
}

}

extern "C" __global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	
	// Index to store
	int index = idx.x + params.image_width * idx.y;

	// Iterate over samples per pixel
	int n = params.spp;

	// Averages
	float4 avg_pixel = make_float4(0.0f);
	float4 avg_normal = make_float4(0.0f);
	float4 avg_diffuse = make_float4(0.0f);

	while (n--) {
		// Pack payload
		RayPacket ray_packet;
		ray_packet.diffuse = make_float3(0.0f);
		ray_packet.normal = make_float3(0.0f);
		ray_packet.throughput = {1.0f, 1.0f, 1.0f};
		ray_packet.value = {0.0f, 0.0f, 0.0f};
		ray_packet.seed = {float(idx.x), float(idx.y), params.time};
		ray_packet.ior = 1.0f;
		ray_packet.depth = 0;

		// Generate ray
		float3 ray_origin;
		float3 ray_direction;
		float radius;

		make_ray(idx, dim, index,
			ray_origin,
			ray_direction,
			ray_packet.seed,
			radius
		);

		unsigned int i0, i1;
		pack_pointer(&ray_packet, i0, i1);
		
		// Launch
		optixTrace(params.handle,
			ray_origin, ray_direction,
			0.0f, 1e16f, 0.0f,
			OptixVisibilityMask(0b11),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0, 0, 0,
			i0, i1
		);

		// Check value
		float4 pixel = make_float4(ray_packet.value);
		float4 normal = make_float4(ray_packet.normal);
		float4 diffuse = make_float4(ray_packet.diffuse);

		// assert(!isnan(pixel.x) && !isnan(pixel.y) && !isnan(pixel.z));
		if (isnan(pixel.x) || isnan(pixel.y) || isnan(pixel.z))
			pixel = {0.0f, 0.0f, 0.0f, 0.0f};

		// Averaged factor
		float factor = filters::box(radius);

		// Accumulate
		avg_pixel += pixel * factor;
		avg_normal += normal * factor;
		avg_diffuse += diffuse * factor;
	}

	// Average
	avg_pixel /= float(params.spp);
	avg_normal /= float(params.spp);
	avg_diffuse /= float(params.spp);

	// Record the results
	params.pbuffer[index] = (avg_pixel + params.pbuffer[index] * params.accumulated)/(params.accumulated + 1);
	params.nbuffer[index] = (avg_normal + params.nbuffer[index] * params.accumulated)/(params.accumulated + 1);
	params.abuffer[index] = (avg_diffuse + params.abuffer[index] * params.accumulated)/(params.accumulated + 1);
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

// Evaluate BRDF of material
__device__ float3 brdf(const Material &mat, float3 n, float3 wi,
		float3 wo, bool entering, Shading out)
{
	// TODO: diffuse should be in conjunction with the material
	if (out & Shading::eTransmission)
		return SpecularTransmission::brdf(mat, n, wi, wo, entering, out);
	
	return mat.diffuse/M_PI + GGX::brdf(mat, n, wi, wo, entering, out);
}

// Evaluate PDF of BRDF
__device__ float pdf(const Material &mat, float3 n, float3 wi,
		float3 wo, bool entering, Shading out)
{
	if (out & Shading::eTransmission)
		return SpecularTransmission::pdf(mat, n, wi, wo, entering, out);
	
	return GGX::pdf(mat, n, wi, wo, entering, out);
}

// Sample BRDF
__device__ float3 sample(const Material &mat, float3 n, float3 wo,
		bool entering, float3 &seed, Shading &out)
{
	if (mat.type & Shading::eTransmission)
		return SpecularTransmission::sample(mat, n, wo, entering, seed, out);

	return GGX::sample(mat, n, wo, entering, seed, out);
}

// Evaluate BRDF: sample, brdf, pdf
template <class BxDF>
__device__ __forceinline__
float3 eval
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	// TODO: pack ags into struct
	wi = sample(mat, n, wo, entering, seed, out);
	if (length(wi) < 1e-6f)
		return make_float3(0.0f);

	pdf = ::pdf(mat, n, wi, wo, entering, out);
	if (pdf < 1e-6f)
		return make_float3(0.0f);

	return brdf(mat, n, wi, wo, entering, out);
}

template <>
__device__ __forceinline__
float3 eval <SpecularTransmission>
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	out = Shading::eTransmission;
	float eta_i = entering ? 1 : mat.refraction;
	float eta_t = entering ? mat.refraction : 1;

	if (dot(n, wo) < 0)
		n = -n;

	float eta = eta_i/eta_t;
	wi = kobra::cuda::Refract(wo, n, eta);
	pdf = 1;

	float fr = kobra::cuda::FrDielectric(dot(n, wi), eta_i, eta_t);
	return make_float3(1 - fr) * (eta * eta)/abs(dot(n, wi));
}

template <>
__device__ __forceinline__
float3 eval <SpecularReflection>
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	float eta_i = entering ? 1 : mat.refraction;
	float eta_t = entering ? mat.refraction : 1;

	if (dot(n, wo) < 0)
		n = -n;

	wi = reflect(-wo, n);
	pdf = 1;

	float fr = kobra::cuda::FrDielectric(dot(n, wi), eta_i, eta_t);
	return make_float3(fr)/abs(dot(n, wi));
}

template <>
__device__ __forceinline__
float3 eval <FresnelSpecular>
(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	float eta_i = entering ? 1 : mat.refraction;
	float eta_t = entering ? mat.refraction : 1;

	float F = kobra::cuda::FrDielectric(dot(wo, n), eta_i, eta_t);

	seed = random3(seed);
	if (fract(seed.x) < F) {
		wi = reflect(-wo, n);
		pdf = F;
		return make_float3(F)/abs(dot(n, wi));
	} else {
		out = Shading::eTransmission;
		float eta = eta_i/eta_t;
		wi = kobra::cuda::Refract(wo, n, eta);
		pdf = 1 - F;
		return make_float3(1 - F) * (eta * eta)/abs(dot(n, wi));
	}
}

// TOdo: union in material for different shading models
__device__ __forceinline__
float3 eval(const Material &mat, float3 n, float3 wo, bool entering,
		float3 &wi, float &pdf, Shading &out, float3 &seed)
{
	if (mat.type & Shading::eTransmission)
		return eval <FresnelSpecular> (mat, n, wo, entering, wi, pdf, out, seed);

	// Fallback to GGX
	return eval <GGX> (mat, n, wo, entering, wi, pdf, out, seed);
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
__device__ float3 sample_area_light(QuadLight light, float3 &seed)
{
	float3 rand = random3(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);
	return light.a + u * light.ab + v * light.ac;
}

__device__ float3 sample_area_light(TriangleLight light, float3 &seed)
{
	float3 rand = random3(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);
	
	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}
	
	return light.a + u * light.ab + v * light.ac;
}

// Check shadow visibility
__device__ bool shadow_visibility(float3 origin, float3 dir, float R)
{
	bool vis = false;
	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	// TODO: max time show be distance to light
	optixTrace(params.handle,
		origin, dir,
		0, R - 0.01f, 0,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
			| OPTIX_RAY_FLAG_DISABLE_ANYHIT
			| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		params.instances, 0, 1,
		j0, j1
	);

	return vis;
}

// Direct lighting for specific types of lights
template <class Light>
__device__ float3 Ld_light(const Light &light, HitGroupData *hit_data, float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	float3 contr_nee {0.0f};
	float3 contr_brdf {0.0f};

	// NEE
	float3 lpos = sample_area_light(light, seed);
	float3 wi = normalize(lpos - x);
	float R = length(lpos - x);

	float3 f = brdf(mat, n, wi, wo, entering, mat.type) * abs(dot(n, wi));

	float ldot = abs(dot(light.normal(), wi));
	if (ldot > 1e-6) {
		float pdf_light = (R * R)/(light.area() * ldot);

		// TODO: how to decide ray type for this?
		float pdf_brdf = pdf(mat, n, wi, wo, entering, mat.type);

		bool vis = shadow_visibility(x, wi, R);
		if (pdf_light > 1e-9 && vis) {
			float weight = power(pdf_light, pdf_brdf);
			float3 intensity = light.intensity;
			contr_nee += weight * f * intensity/pdf_light;
		}
	}

	// BRDF
	Shading out;
	float pdf_brdf;

	f = eval(mat, n, wo, entering, wi, pdf_brdf, out, seed) * abs(dot(n, wi));
	if (length(f) < 1e-6f)
		return contr_nee;

	float pdf_light = 0.0f;

	// TODO: need to check intersection for lights specifically (and
	// arbitrary ones too?)
	float ltime = light.intersects(x, wi);
	if (ltime <= 0.0f)
		return contr_nee;
	
	float weight = 1.0f;
	if (out & eTransmission) {
		return contr_nee;
		// pdf_light = (R * R)/(light.area() * ldot);
	} else {
		R = ltime;
		pdf_light = (R * R)/(light.area() * abs(dot(light.normal(), wi)));
		weight = power(pdf_brdf, pdf_light);
	};

	// TODO: shoot shadow ray up to R
	if (pdf_light > 1e-9 && pdf_brdf > 1e-9) {
		float3 intensity = light.intensity;
		contr_brdf += weight * f * intensity/pdf_brdf;
	}

	return contr_nee + contr_brdf;
}

// Trace ray into scene and get relevant information
__device__ float3 Ld(HitGroupData *hit_data, float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	if (hit_data->n_quad_lights == 0
			&& hit_data->n_tri_lights == 0)
		return float3 {0.0f, 0.0f, 0.0f};

	// TODO: multiply result by # of total lights

	// Random area light for NEE
	random3(seed);
	unsigned int i = seed.x * (hit_data->n_quad_lights + hit_data->n_tri_lights);
	i = min(i, hit_data->n_quad_lights + hit_data->n_tri_lights - 1);

	if (i < hit_data->n_quad_lights) {
		QuadLight light = hit_data->quad_lights[i];
		return Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
	}

	TriangleLight light = hit_data->tri_lights[i - hit_data->n_quad_lights];
	return Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
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
		(HitGroupData *hit_data, uint3 triangle, float2 bary,
		 bool &entering)
{
	float3 e1 = hit_data->vertices[triangle.y] - hit_data->vertices[triangle.x];
	float3 e2 = hit_data->vertices[triangle.z] - hit_data->vertices[triangle.x];
	float3 ng = cross(e1, e2);

	if (dot(ng, optixGetWorldRayDirection()) > 0.0f) {
		ng = -ng;
		entering = false;
	} else {
		entering = true;
	}

	ng = normalize(ng);

	float3 normal = interpolate(hit_data->normals, triangle, bary);
	if (dot(normal, ng) < 0.0f)
		normal = -normal;

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

#define MAX_DEPTH 2

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

	bool entering;
	float3 wo = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit_data, triangle, bary, entering);
	float3 x = interpolate(hit_data->vertices, triangle, bary);

	if (rp->depth == 0) {
		rp->diffuse = material.diffuse;
		rp->normal = n * 0.5f + 0.5f;
	}

	float3 direct = Ld(hit_data, x + 1e-3f * n, wo, n, material, entering, rp->seed);

	// Transfer to payload
	rp->value += rp->throughput * direct;

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
	if (length(f) < 1e-6f)
		return;
	
	float3 T = f * abs(dot(wi, n))/pdf;

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
		offset = 1e-3f * wi;

	// Update ior
	rp->ior = material.refraction;

	// Recurse
	optixTrace(params.handle,
		x + offset, wi,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 0, 0,
		i0, i1
	);
}

extern "C" __global__ void __closesthit__shadow() {}
