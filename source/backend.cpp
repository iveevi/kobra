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

// Create ImageData object from a file
ImageData make_texture(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
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

	// TODO: Copy data to image

	return img;
}

}
