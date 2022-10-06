#ifndef KOBRA_CUDA_INTEROP_H_
#define KOBRA_CUDA_INTEROP_H_

// Engine headers
#include "../backend.hpp"

namespace kobra {

namespace cuda {

static cudaTextureObject_t import_vulkan_texture(const vk::raii::Device &device, const ImageData &img)
{
	// Create a CUDA texture out of the Vulkan image
	cudaExternalMemoryHandleDesc ext_mem_desc {};
	ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	ext_mem_desc.handle.fd = img.get_memory_handle(device);
	ext_mem_desc.size = img.get_size();

	// Import the external memory
	cudaExternalMemory_t tex_mem;
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaImportExternalMemory(&tex_mem, &ext_mem_desc));

	// Create a mipmapped array for the texture
	cudaExternalMemoryMipmappedArrayDesc mip_desc {};
	mip_desc.flags = 0;
	mip_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	mip_desc.numLevels = 1;
	mip_desc.offset = 0;
	mip_desc.extent = make_cudaExtent(
		img.extent.width,
		img.extent.height, 0
	);

	cudaMipmappedArray_t mip_array;
	CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mip_array, tex_mem, &mip_desc));

	// Create the final texture object
	cudaResourceDesc res_desc {};
	res_desc.resType = cudaResourceTypeMipmappedArray;
	res_desc.res.mipmap.mipmap = mip_array;

	cudaTextureDesc tex_desc {};
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	tex_desc.normalizedCoords = true;
	tex_desc.filterMode = cudaFilterModeLinear;

	cudaTextureObject_t tex_obj;
	CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

	return tex_obj;
}

}

}

#endif
