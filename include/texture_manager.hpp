#ifndef KOBRA_TEXTURE_MANAGER_H_
#define KOBRA_TEXTURE_MANAGER_H_

// Standard headers
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

// Engine headers
#include "backend.hpp"

namespace kobra {

// Caches all loaded textures , globally
// TODO: later also allow multihreaded loading
// TODO: remove this class, and put everything in the shared namespace
class TextureManager {
	// TODO: parallel processing with multiple command pools

	// Generic device map
	template <class T>
	using DeviceMap = std::map <vk::Device, T>;

	// Map of image path --> image index
	using ImageMap = std::unordered_map <std::string, size_t>;

	// Map of image path --> image sampler
	using SamplerMap = std::map <std::string, vk::raii::Sampler>;

	// Per device maps
	static DeviceMap <vk::raii::CommandPool>	_command_pools;
	static DeviceMap <ImageMap>			_image_map;
	static DeviceMap <std::vector <ImageData>>	_images;
	static DeviceMap <SamplerMap>			_samplers;
	static DeviceMap <std::mutex>			_mutexes;

	// Create a new command pool for the given device if it doesn't exist yet
	static vk::raii::CommandPool &get_command_pool
			(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &dev) {
		if (_command_pools.find(*dev) == _command_pools.end()) {
			_command_pools.insert({*dev,
				vk::raii::CommandPool {
					dev, {
						vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
						find_graphics_queue_family(phdev)
					}
				}
			});
		}

		return _command_pools.at(*dev);
	}
public:
	// Load a texture
	static const ImageData &load_texture
			(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const std::string &, bool = false);

	// Create a sampler
	static const vk::raii::Sampler &load_sampler
			(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const std::string &);

	// Create an image descriptor for an image
	static vk::DescriptorImageInfo make_descriptor
			(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const std::string &);

	// Bind an image to a descriptor set
	static void bind(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const vk::raii::DescriptorSet &,
			const std::string &,
			uint32_t);
};

}

#endif
