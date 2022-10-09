#ifndef KOBRA_CUDA_RANDOM_H_
#define KOBRA_CUDA_RANDOM_H_

// Engine headers
#include "core.cuh"
#include "math.cuh"

// TODO: namespace
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
unsigned int rand(unsigned int lim, float3 &seed)
{
	uint3 v = *reinterpret_cast <uint3*> (&seed);
	v = pcg3d(v);
	unsigned int r = (v.x + v.y - v.z) % lim;
	*reinterpret_cast <uint3*> (&seed) = v;
	return r;
}

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 random3(float3 &seed)
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
float3 random_sphere(float3 &seed)
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

#endif
