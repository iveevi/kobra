// Vulkan headers
#include <vulkan/vulkan_format_traits.hpp>

// Engine headers
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

		// TODO: separate into own method

		/* TODO: not everything needs to be external...
		img = make_image(*m_device.phdev, *m_device.device,
			m_command_pool, path,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst
				| vk::ImageUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor,
			true
		); */

		// Raw image data
		RawImage raw_image = kobra::load_texture(path);
	
		// Queue to submit commands to
		vk::raii::Queue queue {*m_device.device, 0, 0};

		// Temporary command buffer
		auto cmd = make_command_buffer(*m_device.device, m_command_pool);
	
		// Create the image
		vk::Extent2D extent {
			static_cast <uint32_t> (raw_image.width),
			static_cast <uint32_t> (raw_image.height)
		};

		// Select format
		vk::Format format = vk::Format::eR8G8B8A8Unorm;
		if (raw_image.type == RawImage::RGBA_32_F)
			format = vk::Format::eR32G32B32A32Sfloat;

		img = ImageData(
			*m_device.phdev, *m_device.device,
			format, extent,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst
				| vk::ImageUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor,
			true
		);

		// Copy the image data into a staging buffer
		vk::DeviceSize size = raw_image.width * raw_image.height * vk::blockSize(img.format);

		BufferData buffer {
			*m_device.phdev, *m_device.device, size,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		};

		// Copy the data
		buffer.upload(raw_image.data);

		{
			cmd.begin({});
			img.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

			// Copy the buffer to the image
			copy_data_to_image(cmd,
				buffer.buffer, img.image,
				img.format, raw_image.width, raw_image.height
			);

			// TODO: transition_image_layout should go to the detail namespace...
			img.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
			cmd.end();
		}

		// Submit the command buffer
		queue.submit(
			vk::SubmitInfo {
				0, nullptr, nullptr,
				1, &*cmd
			},
			nullptr
		);

		// Wait
		queue.waitIdle();
	}

	m_images.emplace_back(std::move(img));
	m_image_map[path] = m_images.size() - 1;
	ImageData &ret = m_images.back();

	return ret;
}

// TODO: depracate this function...
vk::raii::Sampler &TextureLoader::load_sampler(const std::string &path)
{
	// TODO: lock guard instead
	if (m_samplers.find(path) != m_samplers.end()) {
		return m_samplers.at(path);
	}

	auto sampler = make_continuous_sampler(*m_device.device);

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
void TextureLoader::bind(const vk::raii::DescriptorSet &dset, const std::string &path, uint32_t binding)
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
