// More vulkan headers
#include <vulkan/vulkan_format_traits.hpp>

// STBI headrs
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "../include/backend.hpp"
#include "../include/core.hpp"

namespace kobra {

// Get (or create) the singleton context
const vk::raii::Context &get_vulkan_context()
{
	// Global context
	static vk::raii::Context context;
	return context;
}

// Copy a buffer to an image
void copy_data_to_image(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Buffer &buffer,
		const vk::raii::Image &image,
		const vk::Format &format,
		uint32_t width,
		uint32_t height)
{
	// Image subresource
	vk::ImageSubresourceLayers subresource {
		vk::ImageAspectFlagBits::eColor,
		0, 0, 1
	};

	// Copy region
	vk::BufferImageCopy region = vk::BufferImageCopy()
		.setBufferOffset(0)
		.setBufferRowLength(width)
		.setBufferImageHeight(height)
		.setImageSubresource(subresource)
		.setImageOffset({ 0, 0, 0 })
		.setImageExtent({ width, height, 1 });

	// Copy buffer to image
	cmd.copyBufferToImage(*buffer, *image,
		vk::ImageLayout::eTransferDstOptimal,
		{region}
	);
}

// Create ImageData object from a file
ImageData make_texture(const vk::raii::CommandBuffer &cmd,
		const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		BufferData &buffer,
		const std::string &filename,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags memory_properties,
		vk::ImageAspectFlags aspect_mask)
{
	// Load the image
	int width;
	int height;
	int channels;

	stbi_set_flip_vertically_on_load(true);
	byte *data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
	KOBRA_ASSERT(data, "Failed to load texture image");

	// Create the image
	vk::Extent2D extent {
		static_cast <uint32_t> (width),
		static_cast <uint32_t> (height)
	};

	ImageData img = ImageData(
		phdev, device,
		vk::Format::eR8G8B8A8Unorm,
		extent,
		tiling,
		usage,
		vk::ImageLayout::ePreinitialized,
		memory_properties,
		aspect_mask
	);

	// Copy the image data into a staging buffer
	vk::DeviceSize size = width * height * vk::blockSize(img.format);

	buffer = BufferData(
		phdev, device,
		size,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Copy the data
	buffer.upload(data, size);

	// First transition the image to the transfer destination layout
	transition_image_layout(cmd,
		*img.image, img.format,
		vk::ImageLayout::ePreinitialized,
		vk::ImageLayout::eTransferDstOptimal
	);

	// Copy the buffer to the image
	copy_data_to_image(cmd,
		buffer.buffer, img.image,
		img.format, width, height
	);

	// Transition the image to the shader read layout
	transition_image_layout(cmd,
		*img.image, img.format,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal
	);

	return img;
}

}
