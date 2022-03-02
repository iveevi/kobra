#ifndef TEXTURE_H_
#define TEXTURE_H_

// Standard headers
#include <fstream>

// STB image loader
#include <stb/stb_image.h>

// Engine headers
#include "buffer_manager.hpp"
#include "core.hpp"
#include "logger.hpp"

namespace mercury {

// Texture structure
struct Texture {
	// Texture data
	uint width;
	uint height;
	uint channels;

	bytes data;
};

// Load texture as byte array
Texture load_image_texture(const std::string &);

// Textures for rasterization
namespace raster {

// Image packet
struct ImagePacket {
	VkImage		image;
	VkDeviceMemory	memory;
	VkImageView	view;
	size_t		width;
	size_t		height;
};

/*
// Create Vulkan image
ImagePacket make_image(const Vulkan::Context &ctx, const Texture &texture, const VkFormat &fmt)
{
	// Create image
	VkImageCreateInfo image_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.extent = { texture.width, texture.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.format = fmt,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
			| VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.flags = 0
	};

	VkImage image;
	VkResult result = vkCreateImage(ctx.device, &image_info, nullptr, &image);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_texture] Failed to create image!");
		return {};
	}

	// Get memory requirements
	VkMemoryRequirements mem_reqs;
	vkGetImageMemoryRequirements(ctx.device, image, &mem_reqs);

	// Allocate memory
	VkMemoryAllocateInfo alloc_info {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = ctx.find_memory_type(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	};

	VkDeviceMemory mem;
	result = vkAllocateMemory(ctx.device, &alloc_info, nullptr, &mem);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_texture] Failed to allocate image memory!");
		return {};
	}

	// Bind memory
	result = vkBindImageMemory(ctx.device, image, mem, 0);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_texture] Failed to bind image memory!");
		return {};
	}

	return {
		.image = image,
		.memory = mem,
		.width = texture.width,
		.height = texture.height
	};
}

// Transition image layout
void transition_image_layout(const Vulkan::Context &ctx,
		const VkCommandPool &cpool,
		const ImagePacket &packet,
		const VkImageLayout &old_layout,
		const VkImageLayout &new_layout)
{
	VkCommandBuffer cmd_buffer = Vulkan::begin_single_time_commands(ctx, cpool);

	VkImageMemoryBarrier barrier {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = packet.image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		},
		.srcAccessMask = 0,
		.dstAccessMask = 0
	};

	// Source layout
	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
			&& new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			&& new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
			&& new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
			| VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	}

	vkCmdPipelineBarrier(cmd_buffer,
		0,
		0,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	Vulkan::submit_single_time_commands(ctx, cpool, cmd_buffer);
}

// Copy buffer to image
void copy_buffer_to_image(const Vulkan::Context &ctx,
		const VkCommandPool &cpool,
		const ImagePacket &packet,
		const VkBuffer &buffer)
{
	VkCommandBuffer cmd_buffer = Vulkan::begin_single_time_commands(ctx, cpool);

	VkBufferImageCopy region {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1
		},
		.imageOffset = { 0, 0, 0 },
		.imageExtent = {
			.width = (uint32_t) packet.width,
			.height = (uint32_t) packet.height,
			.depth = 1
		}
	};

	vkCmdCopyBufferToImage(cmd_buffer,
		buffer,
		packet.image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &region
	);

	Vulkan::submit_single_time_commands(ctx, cpool, cmd_buffer);
}

// Create texture (as VkImage)
VkImage make_texture(const Vulkan::Context &ctx,
		const VkCommandPool &cpool,
		const Texture &texture,
		const VkFormat &fmt)
{
	// Create staging buffer
	BFM_Settings staging_settings {
		.size = texture.data.size(),
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.usage_type = BFM_WRITE_ONLY
	};

	BufferManager <byte> staging_buffer {ctx, staging_settings};

	// Allocate image
	ImagePacket image_packet = make_image(ctx, texture, fmt);

	// Copy data from staging buffer to image
	transition_image_layout(ctx, cpool, image_packet,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
	);

	copy_buffer_to_image(ctx, cpool, image_packet, staging_buffer.vk_buffer());

	transition_image_layout(ctx, cpool, image_packet,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	);

	// TODO: buffer copy method
} */

}

// Textures for ray tracing
namespace raytracing {

// Convert bytes to aligned_vec4 array
Buffer convert_vec4(const Texture &);

// Texture update data
struct TextureUpdate {
	Buffer4f	*textures;
	Buffer4u	*texture_info;

	// Reset indices
	void reset();

	// Write texture data
	void write(const Texture &);

	// Upload texture data
	void upload();
};

}

}

#endif
