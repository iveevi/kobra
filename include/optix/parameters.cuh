#ifndef KOBRA_OPTIX_PARAMETERS_H_
#define KOBRA_OPTIX_PARAMETERS_H_

namespace kobra {

namespace optix {

// Kernel-common parameters for hybrid tracer
struct HT_Parameters {
	// Image resolution
	uint2 resolution;

	// G-buffer textures
	cudaTextureObject_t positions;
	cudaTextureObject_t normals;

	cudaTextureObject_t albedo;
	cudaTextureObject_t specular;
	cudaTextureObject_t extra;

	// Output buffers
	float4 *color_buffer;
};

}

}

#endif
