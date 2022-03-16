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
	// TODO: allow multiple per layer
	Camera			_camera;

	// List of elements
	std::vector <Element>	_elements;

	// Application context
	App::Window		_wctx;

	// Render pass
	VkRenderPass		_render_pass;

	// All rendering pipelines
	struct {
		// VERTEX_TYPE_POSITION
		VkPipeline vertex_position;
	} pipelines;

	// Initialize Vulkan structures
	void _initialize_vulkan_structures(const VkAttachmentLoadOp load) {
		// Create render pass
		_render_pass = _wctx.context.vk->make_render_pass(
			_wctx.context.device,
			_wctx.swapchain,
			load,
			VK_ATTACHMENT_STORE_OP_STORE
		);

		// Load necessary shader modules
		std::vector <VkShaderModule> shaders = _wctx.context.make_shaders({
			"shaders/bin/raster/vtype_position_vert.spv"	// 0: position only
			"shaders/bin/raster/color_frag.spv",		// 1: color
		});

		// Create pipelines
		pipelines.vertex_position = make_pipeline<VERTEX_TYPE_POSITION> (
			_wctx,
			_render_pass,
			shaders[0],
			shaders[1]
		);
	}
public:
	// Default
	Layer() = default;

	// Constructor
	Layer(const App::Window &wctx, const Camera &camera,
			const VkAttachmentLoadOp &load = VK_ATTACHMENT_LOAD_OP_LOAD)
			: _camera(camera), _wctx(wctx) {
		_initialize_vulkan_structures(load);
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
};

}

}

#endif
