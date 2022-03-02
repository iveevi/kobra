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
struct TexturePacket {
	VkImage		image;
	VkDeviceMemory	memory;
	VkImageView	view;
	VkSampler	sampler;

	size_t		width;
	size_t		height;

	// Bind to descriptor set
	void bind(VkDevice device, VkDescriptorSet ds, uint32_t binding) const {
		VkDescriptorImageInfo image_info {
			.sampler = sampler,
			.imageView = view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		VkWriteDescriptorSet descriptor_write {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = ds,
			.dstBinding = binding,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_info
		};

		vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);
	}
};

// Create Vulkan image
inline TexturePacket make_image(const Vulkan::Context &ctx, const Texture &texture, const VkFormat &fmt)
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
	VkResult result = vkCreateImage(ctx.vk_device(), &image_info, nullptr, &image);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_texture] Failed to create image!");
		return {};
	}

	// Get memory requirements
	VkMemoryRequirements mem_reqs;
	vkGetImageMemoryRequirements(ctx.vk_device(), image, &mem_reqs);

	// Allocate memory
	VkMemoryAllocateInfo alloc_info {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = ctx.vk->find_memory_type(ctx.phdev, mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	};

	VkDeviceMemory mem;
	result = vkAllocateMemory(ctx.vk_device(), &alloc_info, nullptr, &mem);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_texture] Failed to allocate image memory!");
		return {};
	}

	// Bind memory
	result = vkBindImageMemory(ctx.vk_device(), image, mem, 0);
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
inline void transition_image_layout(const Vulkan::Context &ctx,
		const VkCommandPool &cpool,
		const TexturePacket &packet,
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
	VkPipelineStageFlags src_stage;
	VkPipelineStageFlags dst_stage;
	
	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
			&& new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			&& new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else {
		Logger::error("[raster::transition_image_layout] Invalid layout transition!");
		return;
	}

	vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	Vulkan::submit_single_time_commands(ctx, cpool, cmd_buffer);
}

// Copy buffer to image
inline void copy_buffer_to_image(const Vulkan::Context &ctx,
		const VkCommandPool &cpool,
		const TexturePacket &packet,
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

// Create texture sampler
inline VkSampler make_sampler(const Vulkan::Context &ctx,
		const VkFilter &filter,
		const VkSamplerAddressMode &address_mode)
{
	VkSamplerCreateInfo sampler_info {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = filter,
		.minFilter = filter,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = address_mode,
		.addressModeV = address_mode,
		.addressModeW = address_mode,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,	// TODO: enable later
		.maxAnisotropy = 16.0f,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkSampler sampler;
	VkResult result = vkCreateSampler(ctx.vk_device(), &sampler_info, nullptr, &sampler);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_sampler] Failed to create sampler!");
		return {};
	}

	return sampler;
}

// Create image view
inline VkImageView make_image_view(const Vulkan::Context &ctx,
		const TexturePacket &packet,
		const VkImageViewType &view_type,
		const VkFormat &format)
{
	VkImageViewCreateInfo view_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = packet.image,
		.viewType = view_type,
		.format = format,
		.components = {
			.r = VK_COMPONENT_SWIZZLE_IDENTITY,
			.g = VK_COMPONENT_SWIZZLE_IDENTITY,
			.b = VK_COMPONENT_SWIZZLE_IDENTITY,
			.a = VK_COMPONENT_SWIZZLE_IDENTITY
		},
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};

	VkImageView view;
	VkResult result = vkCreateImageView(ctx.vk_device(), &view_info, nullptr, &view);
	if (result != VK_SUCCESS) {
		Logger::error("[raster::make_image_view] Failed to create image view!");
		return {};
	}

	return view;
}

// Create texture (as VkImage)
inline TexturePacket make_texture(const Vulkan::Context &ctx,
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
	staging_buffer.write(texture.data.data(), texture.data.size());
	staging_buffer.upload();

	// Allocate image
	TexturePacket image_packet = make_image(ctx, texture, fmt);

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
	
	// Create image view
	image_packet.view = make_image_view(
		ctx, image_packet,
		VK_IMAGE_VIEW_TYPE_2D, fmt
	);

	// Create sampler
	image_packet.sampler = make_sampler(ctx,
		VK_FILTER_LINEAR,
		VK_SAMPLER_ADDRESS_MODE_REPEAT
	);
	
	return image_packet;
}

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
