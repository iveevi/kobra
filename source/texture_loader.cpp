#include "../include/backend.hpp"

namespace kobra {

TextureLoader::TextureLoader(const Device &device)
		: m_device(device)
{
	// Create a command pool for this device
	m_command_pool = vk::raii::CommandPool {
		*m_device.device, {
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			find_graphics_queue_family(*m_device.phdev)
		}
	};
}

// Load a texture
ImageData &TextureLoader::load_texture(const std::string &path)
{
	if (m_image_map.find(path) != m_image_map.end()) {
		size_t index = m_image_map[path];
		return m_images[index];
	}

	// TODO: convert channels to image format
	ImageData img = nullptr;

	if (path == "blank") {
		KOBRA_LOG_FUNC(Log::OK) << "Allocating blank texture\n";

		img = ImageData::blank(*m_device.phdev, *m_device.device);
		img.transition_layout(
			*m_device.device, m_command_pool,
			vk::ImageLayout::eShaderReadOnlyOptimal
		);
	} else{
		KOBRA_LOG_FUNC(Log::OK) << "Loading texture from file: " << path << "\n";

		// TODO: not everything needs to be external...
		img = make_image(*m_device.phdev, *m_device.device,
			m_command_pool, path,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst
				| vk::ImageUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor,
			true
		);
	}

	m_images.emplace_back(std::move(img));
	m_image_map[path] = m_images.size() - 1;
	ImageData &ret = m_images.back();

	return ret;
}

vk::raii::Sampler &TextureLoader::load_sampler(const std::string &path)
{
	// TODO: lock guard instead
	if (m_samplers.find(path) != m_samplers.end()) {
		return m_samplers.at(path);
	}

	auto sampler = make_sampler(
		*m_device.device,
		load_texture(path)
	);

	m_samplers.insert({path, std::move(sampler)});
	vk::raii::Sampler &ret = m_samplers.at(path);
	return ret;
}

vk::DescriptorImageInfo TextureLoader::make_descriptor(const std::string &path)
{
	const vk::raii::Sampler &sampler = load_sampler(path);
	const ImageData &img = load_texture(path);

	return vk::DescriptorImageInfo {
		*sampler,
		*img.view,
		vk::ImageLayout::eShaderReadOnlyOptimal
	};
}

// Bind an image to a descriptor set
void TextureLoader::bind(const vk::raii::DescriptorSet &dset, const std::string &path,uint32_t binding)
{
	auto descriptor = make_descriptor(path);

	vk::WriteDescriptorSet dset_write {
		*dset,
			binding, 0,
			vk::DescriptorType::eCombinedImageSampler,
			descriptor
	};

	m_device.device->updateDescriptorSets(dset_write, nullptr);
}

}
