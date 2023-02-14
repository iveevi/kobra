#ifndef KOBRA_CUDA_RANDOM_H_
#define KOBRA_CUDA_RANDOM_H_

// Engine headers
#include "core.cuh"
#include "math.cuh"

namespace kobra {

namespace cuda {

using Seed = float3 &;

KCUDA_INLINE KCUDA_HOST_DEVICE
uint3 pcg3d(uint3 v)
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

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 pcg3f(Seed seed)
{
	uint3 v = *reinterpret_cast <uint3*> (&seed);
	v = pcg3d(v);
	v &= make_uint3(0x007fffffu);
	v |= make_uint3(0x3f800000u);
	float3 r = *reinterpret_cast <float3*> (&v);
	seed = r - 1.0f;
	return seed;
}

KCUDA_INLINE KCUDA_HOST_DEVICE
float rand_uniform(Seed seed)
{
	seed = pcg3f(seed);
	return fract(seed.x);
}

KCUDA_INLINE KCUDA_HOST_DEVICE
unsigned int rand_uniform(unsigned int lim, Seed seed)
{
	return static_cast <unsigned int> (rand_uniform(seed) * lim);
}

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 rand_uniform_3f(Seed seed)
{
	seed = pcg3f(seed);
	return fract(seed);
}

}

}

#endif
