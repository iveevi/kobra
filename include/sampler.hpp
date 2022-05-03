#ifndef KOBRA_SAMPLER_H_
#define KOBRA_SAMPLER_H_

// Standard headers
#include <map>

// Engine headers
#include "texture.hpp"

namespace kobra {

// Sampler structure
// TODO: remove from TexturePacket
// TODO: separate header?
class Sampler {
	Vulkan::Context _ctx;
	VkImageView	_view = VK_NULL_HANDLE;
	VkSampler	_sampler = VK_NULL_HANDLE;

	// Image info
	size_t		_width = 0;
	size_t		_height = 0;
	VkFormat	_format;

	// Helper function to initialize view and sampler
	void _init(const TexturePacket &);
public:
	// Default constructor
	Sampler() = default;

	// Constructor
	Sampler(const Vulkan::Context &ctx, const TexturePacket &packet,
			const VkImageAspectFlags &aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT)
			: _ctx(ctx), _format(packet.format),
			_width(packet.width), _height(packet.height) {
		// TODO: init method
		_view = make_image_view(
			ctx,
			packet,
			VK_IMAGE_VIEW_TYPE_2D,
			aspect_flags
		);

		_sampler = make_sampler(_ctx,
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_REPEAT
		);

		sampler_cache[_sampler] = this;
	}

	// Constructor
	Sampler(const Vulkan::Context &,
			const VkCommandPool &,
			const std::string &,
			uint32_t = 4);

	// Size in bytes
	size_t bytes() const;

	// Bind sampler to descriptor set
	void bind(const VkDescriptorSet &dset, uint32_t binding) {
		VkDescriptorImageInfo image_info {
			.sampler = _sampler,
			.imageView = _view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		VkWriteDescriptorSet descriptor_write {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = dset,
			.dstBinding = binding,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_info
		};

		vkUpdateDescriptorSets(_ctx.vk_device(), 1, &descriptor_write, 0, nullptr);
	}

	// Get image info
	inline VkDescriptorImageInfo get_image_info() const {
		VkDescriptorImageInfo image_info {
			.sampler = _sampler,
			.imageView = _view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		return image_info;
	}

	// Create a blank sampler
	static Sampler blank_sampler(const Vulkan::Context &ctx, const VkCommandPool &command_pool) {
		static const Texture blank {
			.width = 1,
			.height = 1,
			.channels = 4,
		};

		// Sampler for each unique device
		static std::map <VkDevice, Sampler> sampler_cache;

		if (sampler_cache.find(ctx.vk_device()) == sampler_cache.end()) {
			TexturePacket blank_tp = make_texture(
				ctx, command_pool,
				blank,
				VK_FORMAT_R8G8B8A8_SRGB, // TODO: should match the # of channels in texture
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			);

			blank_tp.transition_manual(ctx, command_pool,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);

			sampler_cache[ctx.vk_device()] = Sampler();
			sampler_cache[ctx.vk_device()]._ctx = ctx;
			sampler_cache[ctx.vk_device()]._init(blank_tp);
		}

		return sampler_cache[ctx.vk_device()];
	}

	// Constructed samplers
	static std::map <VkSampler, const Sampler *> sampler_cache;
};

}

#endif
