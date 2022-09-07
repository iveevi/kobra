#ifndef KOBRA_CUDA_COLOR_H_
#define KOBRA_CUDA_COLOR_H_

// Engine headers
#include "math.cuh"

namespace kobra {

namespace cuda {

__forceinline__
__host__ __device__ float3 to_srgb(const float3& c)
{
	float inv_gamma = 1.0f/2.4f;
	float3 powed = make_float3(
		powf(c.x, inv_gamma),
		powf(c.y, inv_gamma),
		powf(c.z, inv_gamma)
	);

	return make_float3(
		c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
		c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
		c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
	);
}

__forceinline__
__host__ __device__ unsigned char quantize_unsigned_8_bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum {
		N = (1 << 8) - 1,
		Np1 = (1 << 8)
	};

	return (unsigned char) min((unsigned int) (x * (float) Np1), (unsigned int) N);
}

__forceinline__
__host__ __device__ uchar4 make_color(const float3 &c)
{
	float3 srgb = to_srgb(clamp(c, 0.0f, 1.0f));
	return make_uchar4(
		quantize_unsigned_8_bits(srgb.x),
		quantize_unsigned_8_bits(srgb.y),
		quantize_unsigned_8_bits(srgb.z),
		255u
	);
}

__forceinline__
__host__ __device__ uchar4 make_color(const float4 &c)
{
	return make_color(make_float3(c.x, c.y, c.z));
}

}

}

#endif
