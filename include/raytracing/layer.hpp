#ifndef KOBRA_RT_LAYER_H_
#define KOBRA_RT_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "../../shaders/rt/mesh_bindings.h"
#include "../app.hpp"
#include "../backend.hpp"
#include "../camera.hpp"
#include "../layer.hpp"
#include "../logger.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

class Layer : public kobra::Layer <rt::_element> {
	// Private aliases
	using DSLBinding = VkDescriptorSetLayoutBinding;
	using DSLBindings = std::vector <DSLBinding>;

	// All of the layer's cameras
	std::vector <Camera>	_cameras;

	// Active camera
	Camera			*_active_camera;

	// Vulkan context
	Vulkan::Context		_context;

	// Swapchain extent
	VkExtent2D		_extent;

	// Render pass
	VkRenderPass		_render_pass;

	// Pipelines
	struct {
		Vulkan::Pipeline mesh;
		Vulkan::Pipeline postproc;
	} _pipelines;

	// Descriptor set bindings
	static const DSLBindings _mesh_compute_bindings;
	static const DSLBindings _postproc_bindings;

	VkDescriptorSetLayout	_mesh_dsl;
	VkDescriptorSetLayout	_postproc_dsl;

	// Descriptor sets
	VkDescriptorSet		_mesh_ds;
	VkDescriptorSet		_postproc_ds;

	// Initialize mesh compute pipeline
	void _init_mesh_compute_pipeline();
	void _init_postproc_pipeline(const Vulkan::Swapchain &);

	// Initialize all pipelines
	void _init_pipelines(const Vulkan::Swapchain &swapchain,
				const VkDescriptorPool &descriptor_pool) {
		// First, create the DSLs
		_mesh_dsl = _context.vk->make_descriptor_set_layout(
			_context.device,
			_mesh_compute_bindings
		);

		_postproc_dsl = _context.vk->make_descriptor_set_layout(
			_context.device,
			_postproc_bindings
		);

		// Then, create the descriptor sets
		_mesh_ds = _context.vk->make_descriptor_set(
			_context.device,
			descriptor_pool,
			_mesh_dsl
		);

		_postproc_ds = _context.vk->make_descriptor_set(
			_context.device,
			descriptor_pool,
			_postproc_dsl
		);

		// All pipelines
		_init_mesh_compute_pipeline();
		_init_postproc_pipeline(swapchain);
	}

	// Data
	BufferManager <uint>	_pixels;
	BufferManager <uint>	_viewport;
public:
	// Default constructor
	Layer() = default;

	// Constructor
	Layer(const App::Window &wctx)
			: _context(wctx.context), _extent(wctx.swapchain.extent) {
		// Create the render pass
		// TODO: context method
		_render_pass = _context.vk->make_render_pass(
			_context.phdev,
			_context.device,
			wctx.swapchain,
			VK_ATTACHMENT_LOAD_OP_LOAD,
			VK_ATTACHMENT_STORE_OP_STORE,
			false
		);

		// Initialize pipelines
		_init_pipelines(wctx.swapchain, wctx.descriptor_pool);

		// Allocate buffers
		size_t pixels = wctx.swapchain.extent.width
			* wctx.swapchain.extent.height;
		
		BFM_Settings pixel_settings {
			.size = pixels,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_READ_ONLY
		};

		BFM_Settings viewport_settings {
			.size = 2,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		_pixels = BufferManager <uint> (_context, pixel_settings);
		_viewport = BufferManager <uint> (_context, viewport_settings);

		// Fill out the viewport
		_viewport.push_back(_extent.width);
		_viewport.push_back(_extent.height);
		_viewport.sync_size();
		_viewport.upload();

		KOBRA_LOG_FUNC(notify) << "Initialized rt::Layer\n";

		// Bind to descriptor sets
		_pixels.bind(_mesh_ds, MESH_BINDING_PIXELS);
		_viewport.bind(_mesh_ds, MESH_BINDING_VIEWPORT);
		
		_pixels.bind(_postproc_ds, MESH_BINDING_PIXELS);
		_viewport.bind(_postproc_ds, MESH_BINDING_VIEWPORT);
	}

	// Number of cameras
	size_t camera_count() const {
		return _cameras.size();
	}

	// Add a camera to the layer
	void add_camera(const Camera &camera) {
		_cameras.push_back(camera);
	}

	// Active camera
	Camera *active_camera() {
		return _active_camera;
	}

	// Activate a camera
	Camera *activate_camera(size_t index) {
		if (index < _cameras.size()) {
			_active_camera = &_cameras[index];
		} else {
			KOBRA_LOG_FUNC(warn) << "Camera index out of range ["
				<< index << "/" << _cameras.size() << "]";
		}

		return _active_camera;
	}

	// Render
	void render(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) {
		// Bind mesh compute pipeline
		vkCmdBindPipeline(cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			_pipelines.mesh.pipeline
		);

		// Bind descriptor set
		vkCmdBindDescriptorSets(cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			_pipelines.mesh.layout,
			0, 1, &_mesh_ds,
			0, nullptr
		);

		// Dispatch the mesh compute
		vkCmdDispatch(cmd,
			_extent.width,
			_extent.height,
			1
		);

		// Bind post-processing pipeline
		vkCmdBindPipeline(cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipelines.postproc.pipeline
		);

		// Bind descriptor set
		vkCmdBindDescriptorSets(cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipelines.postproc.layout,
			0, 1, &_postproc_ds,
			0, nullptr
		);

		// Begin render pass
		// TODO: context method
		VkRenderPassBeginInfo rp_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = _render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.offset = {0, 0},
				.extent = _extent
			},
			.clearValueCount = 0,
			.pClearValues = nullptr
		};

		vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdDraw(cmd, 6, 1, 0, 0);
		vkCmdEndRenderPass(cmd);
	}
};

}

}

#endif
