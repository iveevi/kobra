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
	VkFormat	format;

	uint32_t	width;
	uint32_t	height;

	// Copy another texture to this one
	// TODO: store image layout
	void copy(const Vulkan::Context &ctx,
			const VkCommandPool &cpool,
			const TexturePacket &tp,
			const VkImageLayout &our_format,
			const VkImageLayout &their_format) {
		// Only same format
		if (tp.format != format) {
			Logger::error() << __PRETTY_FUNCTION__
				<< ": Texture format mismatch" << std::endl;
			return;
		}

		// Copy image
		VkCommandBuffer copy_cmd = Vulkan::begin_single_time_commands(ctx, cpool);

		/* VkImageCopy copy_region = {
			.srcSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
			.srcOffset = { 0, 0, 0 },
			.dstSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
			.dstOffset = { 0, 0, 0 },
			.extent = {
				.width = width,
				.height = height,
				.depth = 1
			}
		}; */

		// Use blit
		VkImageBlit blit = {
			.srcSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
			.srcOffsets = {
				{ 0, 0, 0 },
				{ (int32_t) tp.width, (int32_t) tp.height, 1 }
			},
			.dstSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
			.dstOffsets = {
				{ 0, 0, 0 },
				{ (int32_t) width, (int32_t) height, 1 }
			}
		};

		vkCmdBlitImage(copy_cmd,
			tp.image, their_format,
			image, our_format,
			1, &blit,
			VK_FILTER_LINEAR
		);

		/* vkCmdCopyImage(copy_cmd,
			tp.image, their_format,
			image, our_format,
			1, &copy_region
		); */

		Vulkan::submit_single_time_commands(ctx, cpool, copy_cmd);
	}

	inline void transition_image_layout(const Vulkan::Context &ctx,
			const VkCommandPool &cpool,
			const VkImageLayout &old_layout,
			const VkImageLayout &new_layout) const
	{
		VkCommandBuffer cmd_buffer = Vulkan::begin_single_time_commands(ctx, cpool);

		VkImageMemoryBarrier barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
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
		VkPipelineStageFlags src_stage = 0;
		VkPipelineStageFlags dst_stage = 0;

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
			// Logger::error("[raster::transition_image_layout] Invalid layout transition!");
			// return;

			Logger::warn("[raster::transition_image_layout] Invalid layout transition!");
		}

		vkCmdPipelineBarrier(cmd_buffer,
			src_stage, dst_stage, 0,
			0, nullptr, 0, nullptr,
			1, &barrier
		);

		Vulkan::submit_single_time_commands(ctx, cpool, cmd_buffer);
	}
	
	inline void transition_manual(const Vulkan::Context &ctx,
			const VkCommandPool &cpool,
			const VkImageLayout &old_layout,
			const VkImageLayout &new_layout,
			VkPipelineStageFlags src_stage,
			VkPipelineStageFlags dst_stage) const
	{
		VkCommandBuffer cmd_buffer = Vulkan::begin_single_time_commands(ctx, cpool);

		VkImageMemoryBarrier barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
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

		vkCmdPipelineBarrier(cmd_buffer,
			src_stage, dst_stage, 0,
			0, nullptr, 0, nullptr,
			1, &barrier
		);

		Vulkan::submit_single_time_commands(ctx, cpool, cmd_buffer);
	}
};

// Create Vulkan image
inline TexturePacket make_image(const Vulkan::Context &ctx,
		const Texture &texture,
		const VkFormat &fmt,
		const VkImageUsageFlags &usage_flags)
{
	// Create image
	VkImageCreateInfo image_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = fmt,
		.extent = { texture.width, texture.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage_flags,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
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
		.format = fmt,
		.width = texture.width,
		.height = texture.height
	};
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
		const VkImageViewType &view_type)
{
	VkImageViewCreateInfo view_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = packet.image,
		.viewType = view_type,
		.format = packet.format,
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
// TODO: seriously, save the image layoutttt
inline TexturePacket make_texture(const Vulkan::Context &ctx,
		const VkCommandPool &cpool,
		const Texture &texture,
		const VkFormat &fmt,
		const VkImageUsageFlags &usage_flags,
		const VkImageLayout &layout)
{
	// Allocate image
	TexturePacket packet = make_image(ctx, texture, fmt, usage_flags);

	// Create staging buffer
	if (texture.data.size() > 0) {
		BFM_Settings staging_settings {
			.size = texture.data.size(),
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BufferManager <byte> staging_buffer {ctx, staging_settings};
		staging_buffer.write(texture.data.data(), texture.data.size());
		staging_buffer.upload();

		// Copy data from staging buffer to image
		packet.transition_image_layout(ctx, cpool,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		);

		copy_buffer_to_image(ctx, cpool, packet, staging_buffer.vk_buffer());
	}

	// TODO: buffer copy method
	return packet;
}

// Sampler structure
// TODO: remove from TexturePacket
struct Sampler {
	VkImageView	view;
	VkSampler	sampler;

	// Constructors
	Sampler() = default;
	Sampler(const Vulkan::Context &ctx, const TexturePacket &packet) {
		view = make_image_view(
			ctx,
			packet,
			VK_IMAGE_VIEW_TYPE_2D
		);

		sampler = make_sampler(ctx,
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_REPEAT
		);
	}

	// Bind sampler to pipeline
	void bind(const Vulkan::Context &ctx, VkDescriptorSet ds, uint32_t binding) {
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

		vkUpdateDescriptorSets(ctx.vk_device(), 1, &descriptor_write, 0, nullptr);
	}
};

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
