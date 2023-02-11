#pragma once

#include <glm/glm.hpp>

namespace kobra {

namespace cuda {

__forceinline__ __host__ __device__
float3 to_f3(const glm::vec3 &v)
{
	return make_float3(v.x, v.y, v.z);
}

__forceinline__ __host__ __device__
uint32_t to_ui32(uchar4 v)
{
	// Reversed
	return (v.w << 24) | (v.z << 16) | (v.y << 8) | v.x;
}

}

}
