#include "../../include/gui/layer.hpp"
#include "../../include/gui/rect.hpp"
#include "../../include/gui/sprite.hpp"
#include "../../include/vertex.hpp"

namespace kobra {

namespace gui {

/////////////////////////////
// Static member variables //
/////////////////////////////

const std::vector <DSLB> Layer::_sprites_bindings {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

/////////////////
// Constructor //
/////////////////

Layer::Layer(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const vk::raii::DescriptorPool &descriptor_pool,
		const vk::Extent2D &extent,
		const vk::Format &swapchain_format,
		const vk::Format &depth_format,
		const vk::AttachmentLoadOp &load)
		: _physical_device(phdev), _device(device),
		_command_pool(command_pool),
		_descriptor_pool(descriptor_pool),
		_extent(extent)
{
	// Initialize all Vulkan objects
	_init_vulkan_structures(load, swapchain_format, depth_format);

	// Allocate buffers
	_alloc_rects(_physical_device, _device);

	// Create descriptor set layouts
	_dsl_sprites = make_descriptor_set_layout(_device, _sprites_bindings);

	// Load all shaders
	auto shaders = make_shader_modules(_device, {
		"shaders/bin/gui/basic_vert.spv",
		"shaders/bin/gui/basic_frag.spv",
		"shaders/bin/gui/sprite_vert.spv",
		"shaders/bin/gui/sprite_frag.spv"
	});

	// Pipeline layouts
	_pipelines.shapes_layout = vk::raii::PipelineLayout(
		_device,
		{{}, {}, {}}
	);

	_pipelines.sprites_layout = vk::raii::PipelineLayout(
		_device,
		{{}, *_dsl_sprites, {}}
	);

	// Create pipeline cache
	auto pc = vk::raii::PipelineCache(
		_device,
		vk::PipelineCacheCreateInfo {}
	);

	// Create graphics pipelines
	auto shapes_grp_info = GraphicsPipelineInfo {
		.device = _device,
		.render_pass = _render_pass,

		.vertex_shader = std::move(shaders[0]),
		.fragment_shader = std::move(shaders[1]),

		.vertex_binding = Vertex::vertex_binding(),
		.vertex_attributes = Vertex::vertex_attributes(),

		.pipeline_layout = _pipelines.shapes_layout,
		.pipeline_cache = pc,

		.depth_test = true,
		.depth_write = true
	};

	_pipelines.shapes = make_graphics_pipeline(shapes_grp_info);

	auto sprites_grp_info = GraphicsPipelineInfo {
		.device = _device,
		.render_pass = _render_pass,

		.vertex_shader = std::move(shaders[2]),
		.fragment_shader = std::move(shaders[3]),

		.vertex_binding = Sprite::Vertex::vertex_binding(),
		.vertex_attributes = Sprite::Vertex::vertex_attributes(),

		.pipeline_layout = _pipelines.sprites_layout,
		.pipeline_cache = pc,

		.depth_test = true,
		.depth_write = true
	};

	_pipelines.sprites = make_graphics_pipeline(sprites_grp_info);
}

////////////////////
// Public methods //
////////////////////

void Layer::add_do(const ptr &element)
{
	LatchingPacket lp {
		.layer = this,
	};

	element->latch(lp);
	if (element->type() == Sprite::object_type) {
		_pipeline_map["sprites"].push_back(element);
	} else {
		_pipeline_map["shapes"].push_back(element);
	}
}

}

}
