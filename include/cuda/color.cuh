#ifndef KOBRA_CUDA_COLOR_H_
#define KOBRA_CUDA_COLOR_H_

// Engine headers
#include "math.cuh"

namespace kobra {

namespace cuda {

// HDR to SDR color conversion
__forceinline__ __host__ __device__
float3 srgb(const float3& c)
{
	float inv_gamma = 1.0f/2.4f;
	float3 powed = make_float3(
		powf(c.x, inv_gamma),
		powf(c.y, inv_gamma),
		powf(c.z, inv_gamma)
	);

	float3 s = make_float3(
		c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
		c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
		c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
	);

	return clamp(s, 0.0f, 1.0f);
}

// HDR to LDR with ACES tonemapping
__forceinline__ __host__ __device__
float3 aces(const float3& c)
{
	float3 a = c * (2.51f * c + 0.03f);
	float3 b = c * (2.43f * c + 0.59f) + 0.14f;
	return clamp(a/b, 0.0f, 1.0f);
}

__forceinline__ __host__ __device__
unsigned char quantize_unsigned_8_bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum {
		N = (1 << 8) - 1,
		Np1 = (1 << 8)
	};

	return (unsigned char) min((unsigned int) (x * (float) Np1), (unsigned int) N);
}

__forceinline__ __host__ __device__
uchar4 make_color(const float3 &c, int tonemapping = 0)
{
	float3 s = (tonemapping == 0) ? aces(c) : srgb(c);
	return make_uchar4(
		quantize_unsigned_8_bits(s.x),
		quantize_unsigned_8_bits(s.y),
		quantize_unsigned_8_bits(s.z),
		255u
	);
}

__forceinline__ __host__ __device__
uchar4 make_color(const float4 &c, int tonemapping = 0)
{
	return make_color(make_float3(c.x, c.y, c.z), tonemapping);
}

}

}

#endif
