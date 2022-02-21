#ifndef BUFFER_MANAGER_H_
#define BUFFER_MANAGER_H_

// Engine headers
#include "backend.hpp"
#include "core.hpp"
#include <vulkan/vulkan_core.h>

namespace mercury {

// TODO: alloc subnamespace?
// Usage types for buffer manager
enum BFM_Usage {
	BFM_WRITE_ONLY,
	BFM_READ_ONLY,
	BFM_READ_WRITE,
};

// Settings for buffer manager
struct BFM_Settings {
	size_t			size;
	VkBufferUsageFlags	usage;
	BFM_Usage		usage_type;
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
	const size_t &size() const {
		return settings.size;
	}

	// Get data
	const T *data() const {
		if (settings.usage_type == BFM_READ_ONLY) {
			void *data = context.vk->get_buffer_data(
				context.device,
				gpu_buffer
			);

			return reinterpret_cast <T *> (data);
		}

		return cpu_buffer.data();
	}

	// Get underlying buffer
	const Vulkan::Buffer &buffer() const {
		return gpu_buffer;
	}

	const VkBuffer &vk_buffer() const {
		return gpu_buffer.buffer;
	}

	// Write to buffer (must have write property)
	size_t write(const T *data, size_t size, size_t offset = 0) {
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
	void reset_pushback() {
		push_index = 0;
	}

	void pushback(const T &data) {
		// Don't bother resizing all the time
		if (cpu_buffer.size() <= push_index)
			cpu_buffer.push_back(data);
		else
			cpu_buffer[push_index] = data;
		
		push_index++;
	}

	// Flush cpu buffer to gpu (must have write property)
	void flush() {
		if (settings.usage_type == BFM_WRITE_ONLY) {
			context.vk->map_buffer(
				context.device,
				&gpu_buffer,
				cpu_buffer.data(),
				sizeof(T) * std::min(
					settings.size,
					cpu_buffer.size()
				)
			);
		}
	}

	// Resize buffer
	void resize(size_t size) {
		// Check if resize is needed
		if (size == settings.size)
			return;

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
	}

	// Resize after pushbacks
	void sync_size() {
		resize(cpu_buffer.size());
	}

	// Generating bindings
	VkDescriptorSetLayoutBinding make_dsl_binding(uint32_t binding, VkShaderStageFlags stage) const {
		return VkDescriptorSetLayoutBinding {
			.binding = binding,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = stage,
			.pImmutableSamplers = nullptr
		};
	}

	// Update descriptor set
	void update_descriptor_set(VkDescriptorSet descriptor_set, uint32_t binding) const {
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
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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

}

#endif