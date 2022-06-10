#include "../../include/gui/text.hpp"

namespace kobra {

namespace gui {

////////////////
// Text class //
////////////////

void Text::refresh()
{
	_origin->text(this);
	_origin->add(this);
}

//////////////////////
// TextRender class //
//////////////////////

// Create the pipeline
void TextRender::_make_pipeline(const vk::raii::Device &device, const vk::raii::RenderPass &render_pass)
{
	// Load all shaders
	auto shaders = make_shader_modules(device, {
		"shaders/bin/gui/glyph_vert.spv",
		"shaders/bin/gui/bitmap_frag.spv"
	});

	// Create pipeline layout
	_pipeline_layout = vk::raii::PipelineLayout {
		device,
		{{}, *_descriptor_set_layout, {}}
	};

	auto pipeline_cache = vk::raii::PipelineCache {device, {}};

	// Vertex binding and attribute descriptions
	auto binding_description = Glyph::Vertex::vertex_binding();
	auto attribute_descriptions = Glyph::Vertex::vertex_attributes();

	// Create the graphics pipeline
	auto grp_info = GraphicsPipelineInfo {
		.device = device,
		.render_pass = render_pass,

		.vertex_shader = std::move(shaders[0]),
		.fragment_shader = std::move(shaders[1]),

		.vertex_binding = binding_description,
		.vertex_attributes = attribute_descriptions,

		.pipeline_layout = _pipeline_layout,
		.pipeline_cache = pipeline_cache,

		.depth_test = false,
		.depth_write = false,
	};

	_pipeline = make_graphics_pipeline(grp_info);
}

}

}
