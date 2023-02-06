#ifndef KOBRA_LAYERS_DENOISER_H_
#define KOBRA_LAYERS_DENOISER_H_

// OptiX headers
#include <optix.h>

// Engine headers
#include "../backend.hpp"
#include "../cuda/buffer_data.cuh"

namespace kobra {

namespace layers {

// TODO: multiple backends... (i.e. NRD, Intel, etc)
struct Denoiser {
	// OptiX structures
	OptixDeviceContext context;
	OptixDenoiser denoiser;

	// Image dimensions
	unsigned int width;
	unsigned int height;

	// State and scratch buffers
	cuda::BufferData state;
	cuda::BufferData scratch;

	// Result of the denoiser
	// TODO: keep private, make methods...
	CUdeviceptr result;

	// Options
	enum Guides : uint8_t {
		eNone = 0,
		eNormal = 1,
		eAlbedo = 2,
	} guides;

	// Input for the denoiser
	struct Input {
		CUdeviceptr color = 0;
		CUdeviceptr normal = 0;
		CUdeviceptr albedo = 0;
	};

	// Functions
	static Denoiser make(const vk::Extent2D &, uint8_t = eNone);
};

// Methods
void denoise(Denoiser &, const Denoiser::Input &);

// TODO: resize function

}

}

#endif
