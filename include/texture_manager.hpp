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
			(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &dev,
			const std::string &path) {
		// Get corresponding command pool
		auto &command_pool = get_command_pool(phdev, dev);
		auto &image_map = _image_map[*dev];
		auto &images = _images[*dev];
		auto &mutex = _mutexes[*dev];

		mutex.lock();
		if (image_map.find(path) != image_map.end()) {
			mutex.unlock();
			size_t index = image_map[path];
			return images[index];
		}
		mutex.unlock();

		// TODO: convert channels to image format
		ImageData img = nullptr;

		if (path == "blank") {
			img = ImageData::blank(phdev, dev);
			img.transition_layout(
				dev, command_pool,
				vk::ImageLayout::eShaderReadOnlyOptimal
			);
		} else{
			img = make_image(phdev, dev,
				command_pool, path,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled
					| vk::ImageUsageFlagBits::eTransferDst
					| vk::ImageUsageFlagBits::eTransferSrc,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eColor
			);
		}

		mutex.lock();
		images.emplace_back(std::move(img));
		image_map[path] = images.size() - 1;
		const ImageData &ret = images.back();
		mutex.unlock();

		return ret;
	}

	// Create a sampler
	static const vk::raii::Sampler &load_sampler
			(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &dev,
			const std::string &path) {
		auto &sampler_map = _samplers[*dev];
		auto &mutex = _mutexes[*dev];

		mutex.lock();
		if (sampler_map.find(path) != sampler_map.end()) {
			mutex.unlock();
			return sampler_map.at(path);
		}
		mutex.unlock();

		auto sampler = make_sampler(dev, load_texture(phdev, dev, path));

		mutex.lock();
		sampler_map.insert({path, std::move(sampler)});
		const vk::raii::Sampler &ret = sampler_map.at(path);
		mutex.unlock();

		return ret;
	}

	// Create an image descriptor for an image
	static vk::DescriptorImageInfo make_descriptor
			(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &dev,
			const std::string &path) {
		const vk::raii::Sampler &sampler = load_sampler(phdev, dev, path);
		const ImageData &img = load_texture(phdev, dev, path);

		return vk::DescriptorImageInfo {
			*sampler,
			*img.view,
			vk::ImageLayout::eShaderReadOnlyOptimal
		};
	}

	// Bind an image to a descriptor set
	static void bind(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::DescriptorSet &dset,
			const std::string &path,
			uint32_t binding) {
		auto descriptor = make_descriptor(phdev, device, path);

		vk::WriteDescriptorSet dset_write {
			*dset,
			binding, 0,
			vk::DescriptorType::eCombinedImageSampler,
			descriptor
		};

		device.updateDescriptorSets(dset_write, nullptr);
	}
};

}

#endif
