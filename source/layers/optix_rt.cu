// Engine headers
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/layers/optix_tracer_common.cuh"

extern "C"
{
	__constant__ kobra::Params params;
}

static __forceinline__ __device__ void set_payload(float3 p)
{
	optixSetPayload_0(__float_as_uint( p.x ));
	optixSetPayload_1(__float_as_uint( p.y ));
	optixSetPayload_2(__float_as_uint( p.z ));
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

extern "C" __global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// Map our launch idx to a screen location and create a ray from
	// the camera location through the screen
	float3 ray_origin, ray_direction;
	make_ray(idx, dim, ray_origin, ray_direction);

	// Trace the ray against our scene hierarchy
	unsigned int p0, p1, p2;
	optixTrace(params.handle,
		ray_origin, ray_direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		0, 0, 0,
		p0, p1, p2
	);

	// Our results were packed into opaque 32b registers
	float3 result;
	result.x = __int_as_float(p0);
	result.y = __int_as_float(p1);
	result.z = __int_as_float(p2);

	// Record results in our output raster
	params.image[idx.y * params.image_width + idx.x] = kobra::cuda::make_color(result);
}

extern "C" __global__ void __miss__ms()
{
	// Background color based on ray direction
	// TODO: implement background
	kobra::MissData *miss_data = reinterpret_cast
		<kobra::MissData *> (optixGetSbtDataPointer());

	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z) / (2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y) / M_PI + 0.5f;

	float4 c = tex2D <float4> (miss_data->bg_tex, u, v);

	set_payload(float3 { c.x, c.y, c.z });
}

extern "C" __global__ void __closesthit__ch()
{
	static constexpr float3 colorwheel[] {
		float3 {1, 0, 0},
		float3 {0, 1, 0},
		float3 {0, 0, 1},
		float3 {1, 1, 0},
		float3 {1, 0, 1},
		float3 {0, 1, 1},
		float3 {0.5, 0.5, 0.5},
		float3 {0.5, 1, 0},
		float3 {0, 0.5, 1},
		float3 {1, 0.5, 0},
	};

	static constexpr size_t colorwheel_size = 10;

	// Get data from the SBT
	kobra::HitGroupData *hit_data = reinterpret_cast
		<kobra::HitGroupData *> (optixGetSbtDataPointer());

	// Barycentric coordinates of the hit point
	const float2 bary = optixGetTriangleBarycentrics();

	// Get instance index of the intersected primitive
	int instance_index = optixGetInstanceIndex();
	int primitive_index = optixGetPrimitiveIndex();

	uint3 triangle = hit_data->triangles[primitive_index];

	float2 uv1 = hit_data->texcoords[triangle.x];
	float2 uv2 = hit_data->texcoords[triangle.y];
	float2 uv3 = hit_data->texcoords[triangle.z];

	float2 uv = (1 - bary.x - bary.y) * uv1 + bary.x * uv2 + bary.y * uv3;

	float3 color = hit_data->material.diffuse;
	if (hit_data->textures.has_diffuse) {
		float4 c = tex2D <float4> (hit_data->textures.diffuse, uv.x, uv.y);
		color = float3 { c.x, c.y, c.z };
	}

	set_payload(color);
}
