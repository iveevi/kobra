#ifndef KOBRA_CUDA_COLOR_H_
#define KOBRA_CUDA_COLOR_H_

// Engine headers
#include "cast.cuh"
#include "math.cuh"

namespace kobra {

namespace cuda {

// HDR to SDR color conversion
KCUDA_INLINE KCUDA_HOST_DEVICE
float3 srgb(const float3 &c)
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
KCUDA_INLINE KCUDA_HOST_DEVICE
float3 aces(const float3 &c)
{
	float3 a = c * (2.51f * c + 0.03f);
	float3 b = c * (2.43f * c + 0.59f) + 0.14f;
	return clamp(a/b, 0.0f, 1.0f);
}

KCUDA_INLINE KCUDA_HOST_DEVICE
unsigned char quantize_unsigned_8_bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum {
		N = (1 << 8) - 1,
		Np1 = (1 << 8)
	};

	return (unsigned char) min(
		(unsigned int) (x * (float) Np1),
		(unsigned int) N
	);
}

KCUDA_INLINE KCUDA_HOST_DEVICE
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

template <typename FloatX>
KCUDA_INLINE KCUDA_HOST_DEVICE
uchar4 make_color(const FloatX &c, int tonemapping = 0)
{
	return make_color(make_float3(c.x, c.y, c.z), tonemapping);
}

// HDR color to LDR mapping kernel
enum {
	eTonemappingACES = 0,
	eTonemappingSRGB = 1
};

// CUDA kernel
template <typename FloatX>
inline __global__
void hdr_to_ldr_kernel
		(FloatX *pixels, uint32_t *target,
		int width, int height,
		int tonemapping = eTonemappingACES)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int in_idx = y * width + x;
	int out_idx = (height - y - 1) * width + x;

	FloatX pixel = pixels[in_idx];
	uchar4 color = cuda::make_color(pixel, tonemapping);

	target[out_idx] = cuda::to_ui32(color);
}

// C++ wrapper function
template <typename FloatX>
inline void hdr_to_ldr(FloatX *pixels, uint32_t *target,
		int width, int height,
		int tonemapping = eTonemappingACES)
{
	dim3 block(16, 16);
	dim3 grid(
		(width + block.x - 1) / block.x,
		(height + block.y - 1) / block.y
	);

	hdr_to_ldr_kernel <<<grid, block>>>
		(pixels, target, width, height, tonemapping);
}

}

}

#endif
