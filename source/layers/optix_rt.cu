// Engine headers
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/layers/optix_tracer_common.cuh"

extern "C"
{
	__constant__ kobra::Params params;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
	optixSetPayload_0(__float_as_uint( p.x ));
	optixSetPayload_1(__float_as_uint( p.y ));
	optixSetPayload_2(__float_as_uint( p.z ));
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3 &origin, float3 &direction)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	const float2 d = 2.0f * make_float2(float(idx.x)/dim.x, float(idx.y)/dim.y) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

// Note the __raygen__ prefix which marks this as a ray-generation
// program function
extern "C" __global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// Map our launch idx to a screen location and create a ray from
	// the camera location through the screen
	float3 ray_origin, ray_direction;
	computeRay(idx, dim, ray_origin, ray_direction);

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

__device__ float lerp(uint a, uint b, float t)
{
	return a + (b - a) * t;
}

__device__ float4 lerp(uint4 a, uint4 b, float t)
{
	return make_float4(
		lerp(a.x, b.x, t),
		lerp(a.y, b.y, t),
		lerp(a.z, b.z, t),
		lerp(a.w, b.w, t)
	);
}

__device__ float4 read_tex(cudaTextureObject_t tex, int width, int height, float u, float v)
{
	u = u * width - 0.5f;
	v = v * height - 0.5f;

	int px = floor(u);
	int py = floor(v);

	return make_float4(tex2D <uint4> (tex, u, v));

	float fx = u - px;
	float fy = v - py;

	uint4 c00 = tex2D <uint4> (tex, px, py);
	uint4 c10 = tex2D <uint4> (tex, px + 1, py);
	uint4 c01 = tex2D <uint4> (tex, px, py + 1);
	uint4 c11 = tex2D <uint4> (tex, px + 1, py + 1);

	float4 c0 = lerp(c00, c10, fx);
	float4 c1 = lerp(c01, c11, fx);
	float4 c = lerp(c0, c1, fy);
	return c;
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

	/* float4 c = read_tex(miss_data->bg_tex,
		miss_data->bg_tex_width,
		miss_data->bg_tex_height,
		u, v
	);

	c /= 255.0f; */

	float4 c = tex2D <float4> (miss_data->bg_tex, u, v);

	// uint4 tex = tex2D <uint4> (miss_data->bg_tex, u, v);

	// Compute the color of the background
	// float3 color = 0.5f * ray_direction + 0.5f;
	// color = clamp(color, 0.0f, 1.0f);
	setPayload(float3 { c.x, c.y, c.z });
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

	// Get instance index of the intersected primitive
	int instance_index = optixGetInstanceIndex();

	float3 color = colorwheel[instance_index % colorwheel_size];
	if (instance_index >= hit_data->material_count)
		color = float3 {1, 0, 1};
	else
		color = hit_data->materials[instance_index].diffuse;

	setPayload(color);
}
