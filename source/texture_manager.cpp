#include "../include/texture_manager.hpp"

namespace kobra {

/////////////////////////////
// Static member variables //
/////////////////////////////

TextureManager::DeviceMap <vk::raii::CommandPool>
	TextureManager::_command_pools;
TextureManager::DeviceMap <TextureManager::ImageMap>
	TextureManager::_image_map;
TextureManager::DeviceMap <std::vector <ImageData>>
	TextureManager::_images;
TextureManager::DeviceMap <TextureManager::SamplerMap>
	TextureManager::_samplers;
TextureManager::DeviceMap <std::mutex>
	TextureManager::_mutexes {};

////////////////////
// Static methods //
////////////////////

// Load a texture
const ImageData &TextureManager::load_texture
		(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &dev,
		const std::string &path,
		bool external) {
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
		KOBRA_LOG_FUNC(Log::OK) << "Loading blank texture\n";
		img = ImageData::blank(phdev, dev);
		img.transition_layout(
			dev, command_pool,
			vk::ImageLayout::eShaderReadOnlyOptimal
		);
	} else{
		KOBRA_LOG_FUNC(Log::OK) << "Loading texture from file: " << path << "\n";
		img = make_image(phdev, dev,
			command_pool, path,
			vk::ImageTiling::eLinear,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst
				| vk::ImageUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor,
			external
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
const vk::raii::Sampler &TextureManager::load_sampler
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
vk::DescriptorImageInfo TextureManager::make_descriptor
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
void TextureManager::bind(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::DescriptorSet &dset,
		const std::string &path,
		uint32_t binding) {
	KOBRA_LOG_FUNC(Log::INFO) << "Binding texture: " << path << "\n";
	auto descriptor = make_descriptor(phdev, device, path);

	vk::WriteDescriptorSet dset_write {
		*dset,
		binding, 0,
		vk::DescriptorType::eCombinedImageSampler,
		descriptor
	};

	device.updateDescriptorSets(dset_write, nullptr);
}

}
