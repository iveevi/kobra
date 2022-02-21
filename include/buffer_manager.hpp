#ifndef BUFFER_MANAGER_H_
#define BUFFER_MANAGER_H_

// Engine headers
#include "backend.hpp"
#include "core.hpp"

namespace mercury {

// Manages buffer and associated memory
template <class T>
class BufferManager {
	// TODO: pass vulkan context
	Vulkan		*vk;
	Vulkan::Device	device;

	std::vector <T> cpu_buffer;
	Vulkan::Buffer	gpu_buffer;

	// Constructors
	BufferManager() {}
	BufferManager(Vulkan *ctx, size_t size) {

	}
};

// Aliases
using Buffer4f = BufferManager <aligned_vec4>;

}

#endif