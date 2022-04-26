#include "../include/sampler.hpp"
#include "../include/texture_manager.hpp"
#include "../include/formats.hpp"

namespace kobra {

/////////////////////////////
// Static member variables //
/////////////////////////////

std::map <VkSampler, const Sampler *> Sampler::sampler_cache;

/////////////////////////////
// Sampler private members //
/////////////////////////////

void Sampler::_init(const TexturePacket &packet)
{
	// Create image view
	_view = make_image_view(
		_ctx,
		packet,
		VK_IMAGE_VIEW_TYPE_2D,
		VK_IMAGE_ASPECT_COLOR_BIT
	);

	// Create sampler
	_sampler = make_sampler(_ctx,
		VK_FILTER_LINEAR,
		VK_SAMPLER_ADDRESS_MODE_REPEAT
	);

	// Transfer image info
	_width = packet.width;
	_height = packet.height;
	_format = packet.format;

	// Add to cache
	sampler_cache[_sampler] = this;
}

//////////////////////////
// Sampler constructors //
//////////////////////////

Sampler::Sampler(const Vulkan::Context &ctx,
		const VkCommandPool &command_pool,
		const std::string &path,
		uint32_t channels)
		: _ctx(ctx)
{
	// Load texture at path
	// TODO: auto choose # of channels, instead of passing it here
	Profiler::one().frame("Loading image texture from source");
	// const Texture &texture = load_image_texture(path, channels);
	const Texture &texture = TextureManager::load(path, channels);
	Profiler::one().end();

	// Create texture
	TexturePacket packet = make_texture(_ctx, command_pool,
		texture,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	);

	packet.transition_manual(_ctx, command_pool,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT
	);

	_init(packet);
}

size_t Sampler::bytes() const
{
	return _width * _height * vk_format_table.at(_format).size;
}

}
