// Standard headers
#include <cstdint>

// Engine headers
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/layers/optix_tracer_common.cuh"

using kobra::optix_rt::HitGroupData;
using kobra::optix_rt::MissData;
using kobra::optix_rt::AreaLight;
using kobra::optix_rt::Material;

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

static __forceinline__ __device__ void make_ray(uint3 idx, uint3 dim, float3 &origin, float3 &direction)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;

	int index = idx.x + params.image_width * idx.y;
	float2 d = 2.0f * make_float2(
		float(idx.x + params.xoffset[index])/dim.x,
		float(idx.y + params.yoffset[index])/dim.y
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

// Ray packet data
struct RayPacket {
	float3 throughput;
	float3 value;
	float3 seed;
	int depth;
};

extern "C" __global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// Generate ray
	float3 ray_origin;
	float3 ray_direction;

	make_ray(idx, dim, ray_origin, ray_direction);

	// Pack payload
	RayPacket ray_packet;
	ray_packet.throughput = {1.0f, 1.0f, 1.0f};
	ray_packet.value = {0.0f, 0.0f, 0.0f};
	ray_packet.seed = {float(idx.x), float(idx.y), params.time};
	ray_packet.depth = 0;

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

	// Record the results
	int index = idx.x + params.image_width * idx.y;
	params.pbuffer[index] = (ray_packet.value + params.pbuffer[index] * params.accumulated)/(params.accumulated + 1);
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

__device__ float fract(float x)
{
	return x - floor(x);
}

__device__ float3 fract(float3 x)
{
	return make_float3(
		fract(x.x),
		fract(x.y),
		fract(x.z)
	);
}

__device__ uint3 operator+(uint3 a, unsigned int b)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}

__device__ uint3 operator>>(uint3 a, unsigned int b)
{
	return make_uint3(a.x >> b, a.y >> b, a.z >> b);
}

__device__ uint3 &operator&=(uint3 &a, uint3 b)
{
	a.x &= b.x;
	a.y &= b.y;
	a.z &= b.z;
	return a;
}

__device__ uint3 &operator|=(uint3 &a, uint3 b)
{
	a.x |= b.x;
	a.y |= b.y;
	a.z |= b.z;
	return a;
}

__device__ uint3 &operator^=(uint3 &a, uint3 b)
{
	a.x ^= b.x;
	a.y ^= b.y;
	a.z ^= b.z;
	return a;
}

// Random number generation
__device__ uint3 pcg3d(uint3 v)
{
	v = v * 1664525u + 1013904223u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v ^= v >> 16u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	return v;
}

__device__ unsigned int rand(unsigned int lim)
{
	const uint3 v = pcg3d(make_uint3(
		optixGetLaunchIndex().x,
		optixGetLaunchIndex().y,
		optixGetLaunchIndex().z
	));

	return (v.x + v.y - v.z) % lim;
}

__device__ float3 random3(float3 &seed)
{
	uint3 v = *reinterpret_cast <uint3*> (&seed);
	v = pcg3d(v);
	v &= make_uint3(0x007fffffu);
	v |= make_uint3(0x3f800000u);
	float3 r = *reinterpret_cast <float3*> (&v);
	seed = r - 1.0f;
	return seed;
}

__device__ float3 random_sphere(float3 &seed)
{
	float3 r = random3(seed);
	float ang1 = (r.x + 1.0f) * M_PI;	
	float u = r.y;
	float u2 = u * u;
	
	float sqrt1MinusU2 = sqrt(1.0 - u2);
	
	float x = sqrt1MinusU2 * cos(ang1);
	float y = sqrt1MinusU2 * sin(ang1);
	float z = u;

	return float3 {x, y, z};
}

// GGX microfacet distribution function
__device__ float ggx_d(float3 n, float3 h, Material mat)
{
	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, h), 0.0f, 0.999f));
	
	return (alpha * alpha)
		/ (M_PI * pow(cos(theta), 4)
		* pow(alpha * alpha + tan(theta) * tan(theta), 2.0f));
}

// Smith shadow-masking function (single)
__device__ float G1(float3 n, float3 v, Material mat)
{
	if (dot(v, n) <= 0.0f)
		return 0.0f;

	float alpha = mat.roughness;
	float theta = acos(clamp(dot(n, v), 0.0f, 0.999f));

	float tan_theta = tan(theta);

	float denom = 1 + sqrt(1 + alpha * alpha * tan_theta * tan_theta);
	return 2.0f/denom;
}

// Smith shadow-masking function (double)
__device__ float G(float3 n, float3 wi, float3 wo, Material mat)
{
	return G1(n, wo, mat) * G1(n, wi, mat);
}

// Shlicks approximation to the Fresnel reflectance
__device__ float3 ggx_f(float3 wi, float3 h, Material mat)
{
	float k = pow(1 - dot(wi, h), 5);
	return mat.specular + (1 - mat.specular) * k;
}

// GGX specular brdf
__device__ float3 ggx_brdf(Material mat, float3 n, float3 wi, float3 wo)
{
	if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
		return float3 {0.0f, 0.0f, 0.0f};

	float3 h = normalize(wi + wo);

	float3 f = ggx_f(wi, h, mat);
	float g = G(n, wi, wo, mat);
	float d = ggx_d(n, h, mat);

	float3 num = f * g * d;
	float denom = 4 * dot(wi, n) * dot(wo, n);

	return num / denom;
}

// GGX pdf
__device__ float ggx_pdf(Material mat, float3 n, float3 wi, float3 wo)
{
	if (dot(wi, n) <= 0.0f || dot(wo, n) < 0.0f)
		return 0.0f;

	float3 h = normalize(wi + wo);

	float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
	float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

	float t = 1.0f;
	if (avg_Kd + avg_Ks > 0.0f)
		t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

	float term1 = dot(n, wi)/M_PI;
	float term2 = ggx_d(n, h, mat) * dot(n, h)/(4.0f * dot(wi, h));

	return (1 - t) * term1 + t * term2;
}

// Sample from GGX distribution
__device__ float3 rotate(float3 s, float3 n)
{
	float3 w = n;
	float3 a = float3 {0.0f, 1.0f, 0.0f};

	if (abs(dot(w, a)) > 0.999f)
		a = float3 {0.0f, 0.0f, 1.0f};

	if (abs(dot(w, a)) > 0.999f)
		a = float3 {0.0f, 0.0f, 1.0f};

	float3 u = normalize(cross(w, a));
	float3 v = normalize(cross(w, u));

	return u * s.x + v * s.y + w * s.z;
}

__device__ float3 ggx_sample(float3 n, float3 wo, Material mat, float3 &seed)
{
	float avg_Kd = (mat.diffuse.x + mat.diffuse.y + mat.diffuse.z) / 3.0f;
	float avg_Ks = (mat.specular.x + mat.specular.y + mat.specular.z) / 3.0f;

	float t = 1.0f;
	if (avg_Kd + avg_Ks > 0.0f)
		t = max(avg_Ks/(avg_Kd + avg_Ks), 0.25f);

	float3 r = random3(seed);
	float3 eta = fract(r);

	eta.x = 0.0f;
	if (eta.x < t) {
		// Specular sampling
		float k = sqrt(eta.y/(1 - eta.y));
		float theta = atan(k * mat.roughness);
		float phi = 2.0f * M_PI * eta.z;

		float3 h = float3 {
			sin(theta) * cos(phi),
			sin(theta) * sin(phi),
			cos(theta)
		};

		h = rotate(h, n);

		return reflect(-wo, h);
	}

	// Diffuse sampling
	float theta = acos(sqrt(eta.y));
	float phi = 2.0f * M_PI * eta.z;

	float3 s = float3 {
		sin(theta) * cos(phi),
		sin(theta) * sin(phi),
		cos(theta)
	};

	return rotate(s, n);
}

// BRDF of material
__device__ float3 brdf(Material mat, float3 n, float3 wi, float3 wo)
{
	return ggx_brdf(mat, n, wi, wo) + mat.diffuse/M_PI;
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
		Material mat, float3 &seed)
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

	float3 f = brdf(mat, n, wi, wo) * max(dot(n, wi), 0.0f);

	float ldot = abs(dot(light.normal(), wi));
	if (ldot > 1e-6) {
		float pdf_light = (R * R)/(light.area() * ldot);
		float pdf_brdf = ggx_pdf(mat, n, wi, wo);

		bool vis = shadow_visibility(x, wi, R);
		if (pdf_light > 1e-9 && vis) {
			float weight = power(pdf_light, pdf_brdf);
			float3 intensity = light.intensity;
			contr_nee += weight * f * intensity/pdf_light;
		}
	}

	// BRDF
	wi = ggx_sample(n, wo, mat, seed);
	if (dot(wi, n) <= 0.0f)
		return contr_nee;
	
	f = brdf(mat, n, wi, wo) * max(dot(n, wi), 0.0f);

	float pdf_brdf = ggx_pdf(mat, n, wi, wo);
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
	float3 x = interpolate(hit_data->vertices, triangle, bary)
		+ 1e-3f * n;

	float3 direct = Ld(hit_data, x, wo, n, material, rp->seed);

	// Transfer to payload
	rp->value += direct * rp->throughput;

	// Generate new ray
	float3 wi = ggx_sample(n, wo, material, rp->seed);
	float pdf = ggx_pdf(material, n, wi, wo);

	if (pdf <= 1e-9)
		return;

	float3 f = brdf(material, n, wi, wo) * abs(dot(n, wi));
	float3 T = f/pdf;

	// Russian roulette
	float p = max(rp->throughput.x, max(rp->throughput.y, rp->throughput.z));
	float q = 1 - min(1.0f, p);
	if (fract(rp->seed.x) < q)
		return;

	rp->throughput *= T/(1 - q);
	rp->depth++;

	// Recursive raytrace
	optixTrace(params.handle,
		x, wi,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		0, 0, 0,
		i0, i1
	);
}

extern "C" __global__ void __closesthit__shadow() {}
