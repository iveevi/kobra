#ifndef BUFFER_MANAGER_H_
#define BUFFER_MANAGER_H_

// Engine headers
#include "backend.hpp"
#include "core.hpp"
#include <vulkan/vulkan_core.h>

namespace kobra {

// Usage types for buffer manager
enum BFM_Usage {
	BFM_WRITE_ONLY,
	BFM_READ_ONLY,
	BFM_READ_WRITE,
};

// Settings for buffer manager
struct BFM_Settings {
	size_t			size;
	BFM_Usage		usage_type;
	VkBufferUsageFlags	usage;
	VkDescriptorType	descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
};

// Manages buffer and associated memory
template <class T>
class BufferManager {
private:
	// Members
	Vulkan::Context	context;

	std::vector <T> cpu_buffer;
	Vulkan::Buffer	gpu_buffer;

	size_t		push_index;

	BFM_Settings	settings;

	// Warn null buffer
	void _warn_null_buffer() const {
		if (gpu_buffer.buffer == VK_NULL_HANDLE) {
			KOBRA_LOG_FUNC(warn) << "Buffer not allocated"
				<< " did you forget to initialize"
				<< " the BufferManager?\n";
		}

	}
public:
	// Constructors
	BufferManager() {}
	BufferManager(const Vulkan::Context &vk, const BFM_Settings &s)
			: context(vk), push_index(0), settings(s) {
		// Allocate if requested
		if (settings.usage_type != BFM_READ_ONLY)
			cpu_buffer.resize(settings.size);

		// Create buffer
		context.vk->make_buffer(
			context.phdev,
			context.device,
			gpu_buffer,
			settings.size * sizeof(T),
			settings.usage
		);

		// Add to deletion queue
		context.vk->push_deletion_task(
			[&](Vulkan *vk) {
				vk->destroy_buffer(
					context.device,
					this->gpu_buffer
				);
			}
		);
	}

	// Properties
	size_t bytes() const {
		return settings.size * sizeof(T);
	}

	const size_t &size() const {
		return settings.size;
	}

	const size_t &push_size() const {
		return push_index;
	}

	// Get data
	const T *data() const {
		_warn_null_buffer();
		if (settings.usage_type == BFM_READ_ONLY) {
			void *data = context.vk->get_buffer_data(
				context.device,
				gpu_buffer
			);

			return reinterpret_cast <T *> (data);
		}

		return cpu_buffer.data();
	}

	// Read data from the GPU
	std::vector <T> read() const {
		_warn_null_buffer();
		if (settings.usage_type == BFM_READ_ONLY) {
			std::vector <T> data(settings.size);

			void *data_ptr = context.vk->get_buffer_data(
				context.device,
				gpu_buffer
			);

			memcpy(
				data.data(),
				data_ptr,
				settings.size * sizeof(T)
			);

			return data;
		}

		return cpu_buffer;
	}

	// Get underlying buffer
	const Vulkan::Buffer &buffer() const {
		return gpu_buffer;
	}

	const VkBuffer &vk_buffer() const {
		return gpu_buffer.buffer;
	}

	// Get underlying vector
	const std::vector <T> &vector() const {
		return cpu_buffer;
	}

	// Write to buffer (must have write property)
	size_t write(const T *data, size_t size, size_t offset = 0) {
		_warn_null_buffer();
		if (settings.usage_type == BFM_WRITE_ONLY) {
			// Warn on overflow
			if (offset + size > settings.size) {
				Logger::warn() << "BufferManager::write: write overflow"
					": requested to write " << size << " [+"
					<< offset << "] elements, but buffer size is "
					<< settings.size << " elements" << std::endl;
			}

			// Copy data
			std::memcpy(
				cpu_buffer.data() + offset,
				data,
				size * sizeof(T)
			);
		}

		return 0;
	}

	// Pushback functionality
	void reset_push_back() {
		push_index = 0;
	}

	// Push back overloads
	void push_back(const T &data) {
		// Don't bother resizing all the time
		if (cpu_buffer.size() <= push_index)
			cpu_buffer.push_back(data);
		else
			cpu_buffer[push_index] = data;
		
		push_index++;
	}

	void push_back(const T *data, size_t size) {
		// Don't bother resizing all the time
		if (cpu_buffer.size() <= push_index + size)
			cpu_buffer.resize(push_index + size);

		// Copy data
		std::memcpy(
			cpu_buffer.data() + push_index,
			data,
			size * sizeof(T)
		);

		push_index += size;
	}

	void push_back(const std::vector <T> &data) {
		push_back(data.data(), data.size());
	}

	template <size_t N>
	void push_back(const std::array <T, N> &data) {
		push_back(data.data(), data.size());
	}

	// Push front
	void push_front(const T &data) {
		// Don't bother resizing all the time
		cpu_buffer.insert(cpu_buffer.begin(), data);
		push_index++;
	}

	// Flush cpu buffer to gpu (must have write property)
	void upload() {
		_warn_null_buffer();
		if (settings.usage_type == BFM_WRITE_ONLY) {
			context.vk->map_buffer(
				context.device,
				&gpu_buffer,
				cpu_buffer.data(),
				sizeof(T) * settings.size
			);
		} else {
			KOBRA_LOG_FUNC(warn) << "Calling upload on read-only buffer\n";
		}
	}

	// Resize buffer
	bool resize(size_t size) {
		_warn_null_buffer();

		// Check if resize is needed
		if (size == settings.size)
			return false;

		// Resize cpu buffer
		cpu_buffer.resize(size);

		// Delete old buffer
		context.vk->destroy_buffer(
			context.device,
			gpu_buffer
		);

		// Create new buffer
		context.vk->make_buffer(
			context.phdev,
			context.device,
			gpu_buffer,
			size * sizeof(T),
			settings.usage
		);

		// Update settings
		settings.size = size;

		// Return success
		return true;
	}

	// Resize after pushbacks
	bool sync_size() {
		return resize(cpu_buffer.size());
	}

	// Sync and upload
	void sync_upload() {
		sync_size();
		upload();
	}

	// Clear all data
	void clear() {
		_warn_null_buffer();
		if (settings.usage_type == BFM_WRITE_ONLY) {
			std::memset(
				cpu_buffer.data(),
				0,
				settings.size * sizeof(T)
			);
		}

		// Reset push index and upload
		push_index = 0;
		sync_upload();
	}

	// Update descriptor set
	void bind(VkDescriptorSet descriptor_set, uint32_t binding) const {
		VkDescriptorBufferInfo buffer_info {
			.buffer = gpu_buffer.buffer,
			.offset = 0,
			.range = gpu_buffer.size
		};

		VkWriteDescriptorSet write_descriptor_set {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = nullptr,
			.dstSet = descriptor_set,
			.dstBinding = binding,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = settings.descriptor_type,
			.pImageInfo = nullptr,
			.pBufferInfo = &buffer_info,
			.pTexelBufferView = nullptr
		};

		vkUpdateDescriptorSets(
			context.device.device,
			1, &write_descriptor_set,
			0, nullptr
		);
	}
};

// Aliases
using Buffer4f = BufferManager <aligned_vec4>;
using Buffer4u = BufferManager <aligned_uvec4>;
using Buffer4m = BufferManager <aligned_mat4>;

}

#endif
