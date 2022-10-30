#include "../../include/cuda/alloc.cuh"
#include "../../include/layers/denoiser.cuh"
#include "../../include/optix/core.cuh"

namespace kobra {

namespace layers {

// Create the denoiser layer
Denoiser Denoiser::make(const vk::Extent2D &extent)
{
	// Layer to return
	Denoiser layer;

	// Initialize dimensions
	layer.width = extent.width;
	layer.height = extent.height;

	// Create an OptiX context
	layer.context = optix::make_context();

	// Create the denoiser
	OptixDenoiserOptions denoiser_options = {};
	
	// TODO: options for guide layers, etc
	// denoiser_options.guideAlbedo = 1;
	// denoiser_options.guideNormal = 1;

	OPTIX_CHECK(optixDenoiserCreate(layer.context,
		OPTIX_DENOISER_MODEL_KIND_AOV,
		&denoiser_options,
		&layer.denoiser
	));

	// Optix denoiser sizes
	OptixDenoiserSizes denoiser_sizes;
	OPTIX_CHECK(
		optixDenoiserComputeMemoryResources(
			layer.denoiser,
			layer.width,
			layer.height,
			&denoiser_sizes
		)
	);

	int scratch_size = std::max(
		denoiser_sizes.withOverlapScratchSizeInBytes,
		denoiser_sizes.withoutOverlapScratchSizeInBytes
	);

	// Create the scratch buffers
	layer.state = cuda::BufferData(denoiser_sizes.stateSizeInBytes);
	layer.scratch = cuda::BufferData(scratch_size);

	// Set up denoiser
	OPTIX_CHECK(
		optixDenoiserSetup(layer.denoiser,
			0, layer.width, layer.height,
			layer.state.dev(), layer.state.size(),
			layer.scratch.dev(), layer.scratch.size()
		)
	);

	// Allocate result buffer
	layer.result = cuda::alloc(layer.width * layer.height * sizeof(float4));

	// Return the layer
	return layer;
}

// Perform the denoising
void denoise(Denoiser &layer, const Denoiser::Input &input)
{
	unsigned int row_stride = layer.width * sizeof(float4);

	// All the inputs
	OptixImage2D color_input {
		.data = input.color,
		.width = layer.width,
		.height = layer.height,
		.rowStrideInBytes = row_stride,
		.pixelStrideInBytes = sizeof(float4),
		.format = OPTIX_PIXEL_FORMAT_FLOAT4
	};

	// Output
	OptixImage2D output {
		.data = layer.result,
		.width = layer.width,
		.height = layer.height,
		.rowStrideInBytes = row_stride,
		.pixelStrideInBytes = sizeof(float4),
		.format = OPTIX_PIXEL_FORMAT_FLOAT4
	};

	// Invoke the denoiser
	OptixDenoiserParams denoiser_params = {};
	
	OptixDenoiserGuideLayer denoiser_guide_layer;
	
	OptixDenoiserLayer denoiser_layer;
	denoiser_layer.input = color_input;
	denoiser_layer.output = output;

	// TODO: local CUDA stream?
	OPTIX_CHECK(
		optixDenoiserInvoke(layer.denoiser, 0,
			&denoiser_params,
			layer.state.dev(), layer.state.size(),
			&denoiser_guide_layer, &denoiser_layer,
			1, 0, 0,
			layer.scratch.dev(), layer.scratch.size()
		)
	);
}

}

}
