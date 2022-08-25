#ifndef KOBRA_SHADERS_RANDOM_H_
#define KOBRA_SHADERS_RANDOM_H_

// http://www.jcgt.org/published/0009/03/02/
uvec3 pcg3d(uvec3 v)
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

vec3 random3(inout vec3 seed)
{
	seed = uintBitsToFloat(
		(pcg3d(floatBitsToUint(seed)) & 0x007FFFFFu)
			| 0x3F800000u
	) - 1.0;

	return seed;
}

#endif
