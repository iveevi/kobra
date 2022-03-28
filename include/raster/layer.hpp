#ifndef RASTER_LAYER_H_
#define RASTER_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "pipeline.hpp"
#include "raster.hpp"
#include "../camera.hpp"
#include "../app.hpp"

namespace kobra {

namespace raster {

// Layer class holds
//	all the elements that need
//	to be rendered
class Layer {
	// Layer's camera
	Camera			_camera;

	// List of elements
	// TODO: map elements to their respective pipelines
	std::vector <Element>	_elements;

	// Application context
	App::Window		_wctx;

	// Render pass
	VkRenderPass		_render_pass;

	// All rendering pipelines
	Vulkan::Pipeline	_pipeline;

	// Initialize Vulkan structures
	void _initialize_vulkan_structures(const VkAttachmentLoadOp load) {
		// Create render pass
		_render_pass = _wctx.context.vk->make_render_pass(
			_wctx.context.phdev,
			_wctx.context.device,
			_wctx.swapchain,
			load,
			VK_ATTACHMENT_STORE_OP_STORE,
			true
		);

		// Load necessary shader modules
		std::vector <VkShaderModule> shaders = _wctx.context.make_shaders({
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/color_frag.spv",
			"shaders/bin/raster/blinn_phong_frag.spv"
		});

		// Create pipelines
		_pipeline = make_pipeline (
			_wctx,
			_render_pass,
			shaders[0],
			shaders[2]
		);
	}

	// Initialization status
	bool _initialized = false;
public:
	// Default
	Layer() = default;

	// Constructor
	// TODO: inherit, and add a extra variable to check initialization
	// status
	Layer(const App::Window &wctx, const Camera &camera,
			const VkAttachmentLoadOp &load = VK_ATTACHMENT_LOAD_OP_LOAD)
			: _camera(camera), _wctx(wctx) {
		_initialize_vulkan_structures(load);
		_initialized = true;
	}

	// Get camera
	// TODO: set camera
	Camera &camera() {
		return _camera;
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

	// Render
	void render(const VkCommandBuffer &cmd_buffer, const VkFramebuffer &framebuffer) {
		// Check initialization status
		if (!_initialized) {
			KOBRA_LOG_FUNC(warn) << "calling ::render() on"
				" raster::Layer (" << this << ") which has not"
				" been yet been initialized\n";
		}

		// Start render pass (clear both color and depth)
		VkClearValue clear_colors[] = {
			{.color = {0.0f, 0.0f, 0.0f, 1.0f}},
			{.depthStencil = {1.0f, 0}}
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

		vkCmdBeginRenderPass(cmd_buffer,
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);

		// Bind pipeline
		// TODO: Vulkan::CommandBuffer::bind_pipeline()
		// TODO: pipeline struct with all the necessary info
		vkCmdBindPipeline(cmd_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipeline.pipeline
		);

		// Initialize render packet
		RenderPacket packet {
			.cmd = cmd_buffer,

			.pipeline_layout = _pipeline.layout,

			.view = _camera.view(),
			.proj = _camera.perspective()
		};

		// Render all elements
		for (auto &e : _elements)
			e->render(packet);

		// End render pass
		vkCmdEndRenderPass(cmd_buffer);
	}
};

}

}

#endif
