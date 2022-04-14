#include "../../include/gui/layer.hpp"
#include "../../include/gui/rect.hpp"
#include "../../include/gui/sprite.hpp"

namespace kobra {

namespace gui {

/////////////////////////////
// Static member variables //
/////////////////////////////

const std::vector <Vulkan::DSLB> Layer::_sprites_bindings {
	Vulkan::DSLB {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	}
};

/////////////////
// Constructor //
/////////////////
	
Layer::Layer(const App::Window &wctx, const VkAttachmentLoadOp &load)
		: _wctx(wctx)
{
	// Initialize all Vulkan objects
	_init_vulkan_structures(load);

	// Allocate RenderPacket data
	_alloc_rects();

	// Create descriptor set layouts
	_dsl_sprites = wctx.context.make_dsl(_sprites_bindings);

	// Load all shaders
	auto shaders = _wctx.context.make_shaders({
		"shaders/bin/gui/basic_vert.spv",
		"shaders/bin/gui/basic_frag.spv",
		"shaders/bin/gui/sprite_vert.spv",
		"shaders/bin/gui/sprite_frag.spv"
	});

	// Create pipelines
	Vulkan::PipelineInfo pipeline_info {
		.swapchain = wctx.swapchain,
		.render_pass = _render_pass,

		.dsls = {},

		.vertex_binding = Vertex::vertex_binding(),
		.vertex_attributes = Vertex::vertex_attributes(),

		.depth_test = false,

		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

		.viewport {
			.width = (int) wctx.width,
			.height = (int) wctx.height,
			.x = 0,
			.y = 0
		}
	};

	// Shapes pipeline
	pipeline_info.vert = shaders[0];
	pipeline_info.frag = shaders[1];

	_pipelines.shapes = wctx.context.make_pipeline(pipeline_info);

	// Sprites pipeline
	pipeline_info.vert = shaders[2];
	pipeline_info.frag = shaders[3];

	pipeline_info.vertex_binding = Sprite::Vertex::vertex_binding();
	pipeline_info.vertex_attributes = Sprite::Vertex::vertex_attributes();

	pipeline_info.dsls = {_dsl_sprites};
	_pipelines.sprites = wctx.context.make_pipeline(pipeline_info);
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
