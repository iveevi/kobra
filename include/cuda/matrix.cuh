#ifndef KOBRA_CUDA_MATRIX_H_
#define KOBRA_CUDA_MATRIX_H_

// Engine headers
// TODO: replace with glm?
#include "core.cuh"

namespace kobra {

namespace cuda {

struct mat3 {
	// Column major
	float m[9];

	KCUDA_INLINE KCUDA_HOST_DEVICE
	mat3() {}

	KCUDA_INLINE KCUDA_HOST_DEVICE
	mat3(float3 c1, float3 c2, float3 c3) {
		// Store in column major order
		m[0] = c1.x; m[3] = c2.x; m[6] = c3.x;
		m[1] = c1.y; m[4] = c2.y; m[7] = c3.y;
		m[2] = c1.z; m[5] = c2.z; m[8] = c3.z;
	}
};

KCUDA_INLINE KCUDA_HOST_DEVICE
float3 operator*(mat3 m, float3 v)
{
	return make_float3(
		m.m[0] * v.x + m.m[3] * v.y + m.m[6] * v.z,
		m.m[1] * v.x + m.m[4] * v.y + m.m[7] * v.z,
		m.m[2] * v.x + m.m[5] * v.y + m.m[8] * v.z
	);
}


}

}

#endif
