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
	const float2 d = 2.0f * make_float2(float(idx.x)/dim.x, float(idx.y)/dim.y) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

// Ray packet data
struct RayPacket
{
	float3 value;
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

	unsigned int i0, i1;
	pack_pointer(&ray_packet, i0, i1);
	
	// Launch
	optixTrace(params.handle,
		ray_origin, ray_direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		0, 0, 0,
		i0, i1
	);

	// Unpack payload
	ray_packet = *unpack_point <RayPacket> (i0, i1);

	// Record results in our output raster
	params.image[idx.y * params.image_width + idx.x]
		= kobra::cuda::make_color(ray_packet.value);
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
	rp->value = make_float3(c.x, c.y, c.z);
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

static __forceinline__ __device__ float4 sample_texture
		(HitGroupData *hit_data, cudaTextureObject_t tex, uint3 triangle, float2 bary)
{
	float2 uv1 = hit_data->texcoords[triangle.x];
	float2 uv2 = hit_data->texcoords[triangle.y];
	float2 uv3 = hit_data->texcoords[triangle.z];

	float2 uv = (1 - bary.x - bary.y) * uv1 + bary.x * uv2 + bary.y * uv3;

	return tex2D <float4> (tex, uv.x, uv.y);
}

static __forceinline__ __device__ float3 calculate_normal
		(HitGroupData *hit_data, uint3 triangle, float2 bary)
{
	float3 n1 = hit_data->normals[triangle.x];
	float3 n2 = hit_data->normals[triangle.y];
	float3 n3 = hit_data->normals[triangle.z];

	float3 normal = (1 - bary.x - bary.y) * n1 + bary.x * n2 + bary.y * n3;
	if (dot(normal, optixGetWorldRayDirection()) > 0)
		normal = -normal;
	normal = normalize(normal);

	if (hit_data->textures.has_normal) {
		float4 n4 = sample_texture(hit_data,
			hit_data->textures.normal,
			triangle, bary
		);

		float3 n = 2 * make_float3(n4.x, n4.y, n4.z) - 1;

		// Tangent and bitangent
		float3 t1 = hit_data->tangents[triangle.x];
		float3 t2 = hit_data->tangents[triangle.y];
		float3 t3 = hit_data->tangents[triangle.z];

		float3 b1 = hit_data->bitangents[triangle.x];
		float3 b2 = hit_data->bitangents[triangle.y];
		float3 b3 = hit_data->bitangents[triangle.z];

		float3 tangent = (1 - bary.x - bary.y) * t1 + bary.x * t2 + bary.y * t3;
		float3 bitangent = (1 - bary.x - bary.y) * b1 + bary.x * b2 + bary.y * b3;

		mat3 tbn = mat3(
			normalize(tangent),
			normalize(bitangent),
			normalize(normal)
		);

		normal = normalize(tbn * n);
	}

	return normal;
}

extern "C" __global__ void __closesthit__radiance()
{	
	// Get payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_point <RayPacket> (i0, i1);

	// Get data from the SBT
	HitGroupData *hit_data = reinterpret_cast <HitGroupData *> (optixGetSbtDataPointer());

	if (hit_data->material.type == Shading::eEmissive) {
		rp->value = hit_data->material.diffuse;
		return;
	}

	// Get hit data
	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	uint3 triangle = hit_data->triangles[primitive_index];
	Material material = hit_data->material;

	float3 diffuse = material.diffuse;
	if (hit_data->textures.has_diffuse) {
		diffuse = make_float3(
			sample_texture(hit_data,
				hit_data->textures.diffuse,
				triangle, bary
			)
		);
	}

	float3 color = make_float3(0);

	float3 v = optixGetWorldRayOrigin();
	float3 x = hit_data->vertices[triangle.x] * (1 - bary.x - bary.y) +
		hit_data->vertices[triangle.y] * bary.x +
		hit_data->vertices[triangle.z] * bary.y;
	float3 wi = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit_data, triangle, bary);

	for (int i = 0; i < hit_data->n_area_lights; i++) {
		AreaLight al = hit_data->area_lights[i];

		float3 mid = al.a + (al.ab + al.ac) * 0.5f;

		// Shadow ray
		bool vis = false;
		unsigned int j0, j1;
		pack_pointer <bool> (&vis, j0, j1);

		float3 shadow_origin = x + n * 0.0001f;
		float3 shadow_direction = mid - shadow_origin;

		// TODO: max time show be distance to light
		optixTrace(params.handle_shadow,
			shadow_origin, shadow_direction,
			0, 1e16f, 0,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			5, 0, 1,
			j0, j1
		);

		if (vis)
			color += material.diffuse;
	}

	// color = make_float3(length(x)/20.0f);

	// Transfer to payload
	rp->value = color;
}

extern "C" __global__ void __closesthit__shadow() {}
