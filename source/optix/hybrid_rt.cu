// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/optix/parameters.cuh"
#include "../../include/cuda/math.cuh"

extern "C"
{
	__constant__ kobra::optix::HT_Parameters ht_params;
}

// Ray generation kernel
extern "C" __global__ void __raygen__rg()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

	// Calculate UV coordinates
	const float2 uv = make_float2(
		(float) idx.x/(float) ht_params.resolution.x,
		(float) idx.y/(float) ht_params.resolution.y
	);

	// Store color
	ht_params.color_buffer[index] = 0.5f + 0.5f * tex2D <float4>
		(ht_params.normals, uv.x, 1 - uv.y);
}

// Closest hit kernel
extern "C" __global__ void __closesthit__ch()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

	// Store color
	ht_params.color_buffer[index] = {1, 0, 0, 1};
}

// Miss kernel
extern "C" __global__ void __miss__ms()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * ht_params.resolution.x;

	// Store color
	ht_params.color_buffer[index] = {0, 0, 1, 1};
}
