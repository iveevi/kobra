#ifndef LAYER_H_
#define LAYER_H_

// Standard headers
#include <map>
#include <vector>

// Engine headers
#include "../app.hpp"
#include "../layer.hpp"
#include "gui.hpp"
#include "text.hpp"

namespace kobra {

namespace gui {

// Contains a set of
//	GUI elements to render
// TODO: derive from element?
class Layer : public kobra::Layer <_element> {
	// Private aliases
	template <class T>
	using str_map = std::map <std::string, T>;

	// Element arranged by pipeline
	str_map <std::vector <ptr>>	_pipeline_map;

	// Set of Text Render objects
	// for each font
	std::vector <TextRender>	_text_renders;

	// Map of font names/aliases to
	//	their respective TextRender indices
	str_map <int>			_font_map;

	// Application context
	App::Window			_wctx;

	// Vulkan structures
	VkRenderPass			_render_pass;

	// Pipelines
	struct {
		Vulkan::Pipeline	shapes;
		Vulkan::Pipeline	sprites;
	} _pipelines;

	// Descriptor set layouts
	Vulkan::DSL			_dsl_sprites;

	// Descriptor set layout bindings
	static const std::vector <Vulkan::DSLB>	_sprites_bindings;

	// Allocation methods
	void _init_vulkan_structures(VkAttachmentLoadOp load) {
		// Create render pass
		_render_pass = _wctx.context.vk->make_render_pass(
			_wctx.context.phdev,
			_wctx.context.device,
			_wctx.swapchain,
			load,
			VK_ATTACHMENT_STORE_OP_STORE
		);
	}

	// Hardware resources
	struct {
		VertexBuffer vb;
		IndexBuffer ib;
	} rects;

	// Allocation methods
	void _alloc_rects() {
		BFM_Settings vb_settings {
			.size = 1024,
			.usage_type = BFM_WRITE_ONLY,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		};

		BFM_Settings ib_settings {
			.size = 1024,
			.usage_type = BFM_WRITE_ONLY,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		};

		rects.vb = VertexBuffer(_wctx.context, vb_settings);
		rects.ib = IndexBuffer(_wctx.context, ib_settings);
	}
public:
	// Default
	Layer() = default;

	// Constructor
	// TODO: _layer base class
	Layer(const App::Window &, const VkAttachmentLoadOp & = VK_ATTACHMENT_LOAD_OP_LOAD);

	// Add action
	void add_do(const ptr &) override;

	// Serve descriptor sets
	Vulkan::DS serve_sprite_ds() {
		return _wctx.context.make_ds(_wctx.descriptor_pool, _dsl_sprites);
	}

	// Adding a scene
	void add_scene(const Scene &scene) override {
		KOBRA_LOG_FUNC(warn) << "Not implemented\n";
	}

	// Load fonts
	void load_font(const std::string &alias, const std::string &path) {
		size_t index = _text_renders.size();
		_text_renders.push_back(TextRender(_wctx, _render_pass, path));
		_font_map[alias] = index;
	}

	// Get TextRender
	TextRender *text_render(int index) {
		return &_text_renders[index];
	}

	TextRender *text_render(const std::string &alias) {
		return text_render(_font_map.at(alias));
	}

	// Render using command buffer and framebuffer
	void render(const VkCommandBuffer &cmd_buffer, const VkFramebuffer &framebuffer) {
		// Start render pass
		// TODO: vulkan method
		VkClearValue clear_colors[] {
			{0.0f, 0.0f, 0.0f, 1.0f}, // Color
			{1.0f, 0.0f, 0.0f, 1.0f}  // Depth
		};

		VkRenderPassBeginInfo render_pass_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = _render_pass,
			.framebuffer = framebuffer,
			.renderArea {
				.offset = {0, 0},
				.extent = _wctx.swapchain.extent
			},
			.clearValueCount = 2,
			.pClearValues = clear_colors
		};

		vkCmdBeginRenderPass(cmd_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);


		// Initialize render packet
		RenderPacket rp {
			.cmd = cmd_buffer,
			.sprite_layout = _pipelines.sprites.layout,
		};

		// Render all plain shapes
		vkCmdBindPipeline(cmd_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipelines.shapes.pipeline
		);

		for (auto &e : _pipeline_map["shapes"])
			e->render_element(rp);

		// Render all sprites
		vkCmdBindPipeline(cmd_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipelines.sprites.pipeline
		);

		for (auto &e : _pipeline_map["sprites"])
			e->render_element(rp);

		// Render all the text renders
		for (auto &tr : _text_renders)
			tr.render(_wctx.context, _wctx.command_pool, cmd_buffer);

		// End render pass
		vkCmdEndRenderPass(cmd_buffer);
	}
};

}

}

#endif
