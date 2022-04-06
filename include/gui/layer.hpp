#ifndef LAYER_H_
#define LAYER_H_

// Standard headers
#include <map>
#include <vector>

// Engine headers
#include "../app.hpp"
#include "gui.hpp"
#include "text.hpp"

namespace kobra {

namespace gui {

// Contains a set of
//	GUI elements to render
// TODO: derive from element?
class Layer {
	// All elements to render
	std::vector <Element>		_elements;

	// Set of Text Render objects
	// for each font
	std::vector <TextRender>	_text_renders;

	// Map of font names/aliases to
	//	their respective TextRender indices
	std::map <std::string, int>	_font_map;

	// Application context
	App::Window			_wctx;

	// Vulkan structures
	VkRenderPass			_render_pass;

	// Pipelines
	// TODO: struct
	Vulkan::Pipeline			_grp_shapes;

	// Allocation methods
	void _init_vulkan_structures(VkAttachmentLoadOp load) {
		// Create render pass
		_render_pass = _wctx.context.vk->make_render_pass(
			_wctx.context.phdev,
			_wctx.context.device,
			_wctx.swapchain,
			load,
			VK_ATTACHMENT_STORE_OP_STORE,
			true
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
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BFM_Settings ib_settings {
			.size = 1024,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		rects.vb = VertexBuffer(_wctx.context, vb_settings);
		rects.ib = IndexBuffer(_wctx.context, ib_settings);
	}
public:
	// Default
	Layer() = default;

	// Constructor
	// TODO: _layer base class
	Layer(const App::Window &wctx, const VkAttachmentLoadOp &load = VK_ATTACHMENT_LOAD_OP_LOAD)
			: _wctx(wctx) {
		// Initialize all Vulkan objects
		_init_vulkan_structures(load);

		// Allocate RenderPacket data
		_alloc_rects();

		// Load all shaders
		auto shaders = _wctx.context.make_shaders({
			"shaders/bin/gui/basic_vert.spv",
			"shaders/bin/gui/basic_frag.spv"
		});
	
		// Create pipelines
		Vulkan::PipelineInfo grp_info {
			.swapchain = wctx.swapchain,
			.render_pass = _render_pass,
			
			.vert = shaders[0],
			.frag = shaders[1],
			
			.dsls = {},

			.vertex_binding = Vertex::vertex_binding(),
			.vertex_attributes = Vertex::vertex_attributes(),

			.depth_test = true,

			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

			.viewport {
				.width = (int) wctx.width,
				.height = (int) wctx.height,
				.x = 0,
				.y = 0
			}
		};

		_grp_shapes = wctx.context.make_pipeline(grp_info);
	}

	// Add elements
	void add(const Element &element) {
		_elements.push_back(element);
	}

	void add(_element *ptr) {
		_elements.push_back(Element(ptr));
	}

	// Add multiple elements
	void add(const std::vector <Element> &elements) {
		_elements.insert(
			_elements.end(),
			elements.begin(),
			elements.end()
		);
	}

	void add(const std::vector <_element *> &elements) {
		for (auto &e : elements)
			_elements.push_back(Element(e));
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
		VkClearValue clear_color = {0.0f, 0.0f, 0.0f, 1.0f};

		VkRenderPassBeginInfo render_pass_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = _render_pass,
			.framebuffer = framebuffer,
			.renderArea {
				.offset = {0, 0},
				.extent = _wctx.swapchain.extent
			},
			.clearValueCount = 1,
			.pClearValues = &clear_color
		};

		vkCmdBeginRenderPass(cmd_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

		// Bind graphics pipeline
		vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _grp_shapes.pipeline);

		// Initialize render packet
		RenderPacket rp {
			.rects {
				.vb = &rects.vb,
				.ib = &rects.ib
			}
		};

		// Reset RenderPacket
		rp.reset();

		// Render all elements onto the RenderPacket
		for (auto &elem : _elements)
			elem->render_element(rp);

		// Sync RenderPacket
		rp.sync();

		// Render all parts of the RenderPacket
		// TODO: separate method

		// Draw rectangles
		VkBuffer	vbuffers[] = {rects.vb.vk_buffer()};
		VkDeviceSize	offsets[] = {0};

		vkCmdBindVertexBuffers(cmd_buffer, 0, 1, vbuffers, offsets);
		vkCmdBindIndexBuffer(cmd_buffer, rects.ib.vk_buffer(), 0, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(cmd_buffer, rects.ib.push_size(), 1, 0, 0, 0);

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
