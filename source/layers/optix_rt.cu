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

using kobra::optix_rt::PathSample;
using kobra::optix_rt::Reservoir;

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
	float3	diffuse;
	float3	position;
	float3	normal;
	float3	throughput;
	float3	value;
	float3	seed;
	float	ior;
	int	depth;
	int	imgidx;
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

__forceinline__ __device__
float4 nan_fix(float4 v)
{
	return make_float4(
		isnan(v.x) ? 0.0f : v.x,
		isnan(v.y) ? 0.0f : v.y,
		isnan(v.z) ? 0.0f : v.z,
		isnan(v.w) ? 0.0f : v.w
	);
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

	// Reset the reservoir if needed
	if (params.accumulated == 0) {
		params.reservoirs[index].reset();
		params.prev_reservoirs[index].reset();
		params.spatial_reservoirs[index].reset();
		params.prev_spatial_reservoirs[index].reset();
		params.sampling_radius[index] = min(
			params.image_width,
			params.image_height
		)/10.0f;
	}

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
		ray_packet.imgidx = index;

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

		// nan_fix(pixel);
		// nan_fix(normal);
		// nan_fix(diffuse);

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

#define PROGRESSIVE

#ifdef PROGRESSIVE
	params.pbuffer[index] = (avg_pixel + params.pbuffer[index] * params.accumulated)/(params.accumulated + 1);
	params.nbuffer[index] = (avg_normal + params.nbuffer[index] * params.accumulated)/(params.accumulated + 1);
	params.abuffer[index] = (avg_diffuse + params.abuffer[index] * params.accumulated)/(params.accumulated + 1);
#else
	// Store
	params.pbuffer[index] = avg_pixel;
	params.nbuffer[index] = avg_normal;
	params.abuffer[index] = avg_diffuse;
#endif
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
		return FresnelSpecular::brdf(mat, n, wi, wo, entering, out);
	
	return mat.diffuse/M_PI + GGX::brdf(mat, n, wi, wo, entering, out);
}

// Evaluate PDF of BRDF
__device__ float pdf(const Material &mat, float3 n, float3 wi,
		float3 wo, bool entering, Shading out)
{
	if (out & Shading::eTransmission)
		return FresnelSpecular::pdf(mat, n, wi, wo, entering, out);
	
	return GGX::pdf(mat, n, wi, wo, entering, out);
}

// Sample BRDF
__device__ float3 sample(const Material &mat, float3 n, float3 wo,
		bool entering, float3 &seed, Shading &out)
{
	if (mat.type & Shading::eTransmission)
		return FresnelSpecular::sample(mat, n, wo, entering, seed, out);

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
// #define LIGHT_SAMPLES 5

#ifdef LIGHT_SAMPLES

	float3 contr {0.0f};

	for (int k = 0; k < LIGHT_SAMPLES; k++) {
		random3(seed);
		unsigned int i = seed.x * (hit_data->n_quad_lights + hit_data->n_tri_lights);
		i = min(i, hit_data->n_quad_lights + hit_data->n_tri_lights - 1);

		if (i < hit_data->n_quad_lights) {
			QuadLight light = hit_data->quad_lights[i];
			contr += Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
		} else {
			TriangleLight light = hit_data->tri_lights[i - hit_data->n_quad_lights];
			contr += Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
		}
	}

	return contr/LIGHT_SAMPLES;

#else 

	random3(seed);
	unsigned int i = seed.x * (hit_data->n_quad_lights + hit_data->n_tri_lights);
	i = min(i, hit_data->n_quad_lights + hit_data->n_tri_lights - 1);

	if (i < hit_data->n_quad_lights) {
		QuadLight light = hit_data->quad_lights[i];
		return Ld_light(light, hit_data, x, wo, n, mat, entering, seed);
	}

	TriangleLight light = hit_data->tri_lights[i - hit_data->n_quad_lights];
	return Ld_light(light, hit_data, x, wo, n, mat, entering, seed);

#endif

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

#define MAX_DEPTH 3

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

	// Calculate relevant data for the hit
	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	uint3 triangle = hit_data->triangles[primitive_index];

	Material material = hit_data->material;

	// TODO: check for light, not just emissive material
	if (hit_data->material.type == Shading::eEmissive) {
		rp->value += rp->throughput * material.emission;
		rp->throughput = {0, 0, 0};

		rp->position = optixGetWorldRayOrigin();
		rp->normal = {0, 0, 0};
		rp->diffuse = material.emission;

		return;
	}
	
	calculate_material(hit_data, material, triangle, bary);

	bool entering;
	float3 wo = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit_data, triangle, bary, entering);
	float3 x = interpolate(hit_data->vertices, triangle, bary);

	float3 direct = Ld(hit_data, x + 1e-3f * n, wo, n, material, entering, rp->seed);

	// Transfer to payload
	bool primary = (rp->depth == 0);

	if (primary) {
		rp->normal = n;
		rp->diffuse = material.diffuse;
	}

	float3 cT = rp->throughput;

	bool restir_mode = (
		primary
		&& params.options.use_reservoir
		// && material.type != Shading::eTransmission
	);

	if (!restir_mode)
		rp->value += cT * direct;

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
	if (length(f) < 1e-6f)
		return;

	/* Resampling
	if (rp->depth == 0) {
		Reservoir r = params.reservoirs[rp->imgidx];
		float weight = pdf;
		kobra::optix_rt::DirectionSample ds = {wi, rp->seed};
		r.update(ds, weight);
		ds = r.sample;
		wi = ds.dir;
		rp->seed = ds.seed;
	} */
	
	float3 T = f * abs(dot(wi, n))/pdf;

	// Russian roulette
	float p = max(rp->throughput.x, max(rp->throughput.y, rp->throughput.z));
	float q = 1 - min(1.0f, p);

	// if (fract(rp->seed.x) < q)
	//	return;

	if (!(primary && params.options.use_reservoir))
		rp->throughput *= T;
		// rp->throughput *= T/(1 - q);

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

	// Resampling
	if (restir_mode) {
		const float max_radius = min(params.image_width, params.image_height);

		float &sampling_radius = params.sampling_radius[rp->imgidx];

		// Get the this pixel's reservoirs
		Reservoir &temporal = params.reservoirs[rp->imgidx];
		Reservoir &spatial = params.spatial_reservoirs[rp->imgidx];

		// TODO: double buffer spatial reservoirs if theyre count is low

		// Reset temporal reservoir every 10 samples
		//	to avoid stagnation of samples
		/* if (temporal.count >= 30)
			temporal.reset();

		// Also reset spatial reservoir
		if (spatial.count >= 500)
			spatial.reset(); */

		// Populate temporal reservoir
		float weight = max(rp->value)/pdf;
		
		float depth = length(optixGetWorldRayOrigin() - x);

		// TODO: The ray misses if its depth is 1
		//	but not if it hits a light (check value)
		//	this fixes lights being black with ReSTIR
		bool missed = (rp->depth == 1);

		PathSample ps = {
			.value = rp->value,
			.dir = wi,
			.miss = missed,

			.v_normal = n,
			.v_position = x,
			
			.s_normal = rp->normal,
			.s_position = rp->position,
			
			.brdf = f * abs(dot(wi, n))/pdf,
			.pdf = pdf,
			.depth = depth,
		};

		temporal.update(ps, weight);
		temporal.W = temporal.weight
			/(temporal.count * max(temporal.sample.value) + 1e-6f);

		// Spatial reservoir resampling
		int Z = 0;

		int idx = rp->imgidx % params.image_width;
		int idy = rp->imgidx / params.image_width;

		const int SPATIAL_SAMPLES = (spatial.count < 250) ? 9 : 3;

		int empty_res = 0;

		int success = 0;
		for (int i = 0; i < SPATIAL_SAMPLES; i++) {
			random3(rp->seed);
			float3 random = fract(random3(rp->seed));

			float radius = random.x * sampling_radius;
			float theta = random.y * 2.0f * M_PI;

			int nx = idx + radius * cos(theta);
			int ny = idy + radius * sin(theta);

			if (nx < 0 || nx >= params.image_width
				|| ny < 0 || ny >= params.image_height)
				continue;

			int idx = nx + ny * params.image_width;

			// Skip if corresponding reservoir is empty
			Reservoir *s;

			if (params.prev_spatial_reservoirs[idx].count > 50)
				s = &params.prev_spatial_reservoirs[idx];
			else
				s = &params.reservoirs[idx];

			if (s->count == 0 || params.accumulated == 0) {
				empty_res++;
				continue;
			}

			// Check geometric similarity
			float3 sn = s->sample.v_normal;
			float3 sx = s->sample.v_position;

			float angle = 180 * acos(dot(sn, n))/M_PI;
			float ndepth = abs(s->sample.depth - depth)/depth;

// #define HIGHLIGHT_SPATIAL

			if (angle > 10 || ndepth > 0.1f) {
#ifdef HIGHLIGHT_SPATIAL
				rp->value = {1, 1, 0};
				return;
#endif

				continue;
			}

			// Merge reservoirs if the sample point can be connected
			float R = length(s->sample.s_position - x);
			float3 dir = normalize(s->sample.s_position - x);

			bool vis = false;
			if (s->sample.miss) {
				dir = s->sample.dir;
				vis = shadow_visibility(x + dir * 1e-3f, dir, 1e6);
			} else {
				vis = shadow_visibility(x + dir * 1e-3f, dir, R);
			}

			if (vis) {
#ifdef HIGHLIGHT_SPATIAL
				rp->value = {0, 1, 0};
				return;
#endif

				float3 x1q = s->sample.v_position;
				float3 x1r = rp->position;
				float3 x2q = s->sample.s_position;

				float3 v1q2r = x1q - x2q;
				float3 v1r2q = x1r - x2q;

				float d1q2q = length(v1q2r);
				float d1r2q = length(v1r2q);

				v1r2q /= d1r2q;
				v1q2r /= d1q2q;

				float3 sample_n = s->sample.s_normal;
				float phi_s = acos(dot(sample_n, v1r2q));
				float phi_r = acos(dot(sample_n, v1q2r));

				float J = abs(phi_r/phi_s) * (d1q2q * d1q2q)/(d1r2q * d1r2q);

				spatial.merge(*s, max(s->sample.value)/J);
				Z += s->count;

				success++;
			} else {
#ifdef HIGHLIGHT_SPATIAL
				rp->value = {1, 0, 1};
				return;
#endif
			}
		}

#ifdef HIGHLIGHT_SPATIAL
		if (empty_res == SPATIAL_SAMPLES) {
			rp->value = {1, 0, 0};
			return;
		}

		if (Z == 0) {
			rp->value = {0, 0, 1};
			return;
		}
#endif

		if (success == 0 && params.accumulated > 0)
			sampling_radius = max(sampling_radius * 0.5f, 3.0f);

		Z += temporal.count;
		spatial.W = spatial.weight/(Z * max(spatial.sample.value) + 1e-6f);

		// Compute value
		float3 tvalue = direct + temporal.sample.brdf
			* temporal.sample.value;

		float3 svalue = direct + T
			* spatial.sample.value;

		bool specural = (material.type == Shading::eTransmission)
			|| (material.roughness < 0.05f);

		if (specural)
			rp->value = tvalue;
		else
			rp->value = svalue;

		// Double buffering
		params.prev_reservoirs[rp->imgidx] = temporal;
		params.prev_spatial_reservoirs[rp->imgidx] = spatial;
	}
	
	rp->diffuse = material.diffuse;
	rp->normal = n;
	rp->position = x;
}

extern "C" __global__ void __closesthit__shadow() {}
