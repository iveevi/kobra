#ifndef RASTER_LAYER_H_
#define RASTER_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "../../shaders/raster/bindings.h"
#include "../app.hpp"
#include "../buffer_manager.hpp"
#include "../camera.hpp"
#include "../layer.hpp"
#include "../scene.hpp"
#include "mesh.hpp"
#include "raster.hpp"

namespace kobra {

namespace raster {

// Layer class holds
//	all the elements that need
//	to be rendered
class Layer : public kobra::Layer <raster::_element> {
public:
	// Rendering mode
	enum class Mode {
		ALBEDO,
		NORMAL,
		BLINN_PHONG
	};
protected:
	// Private aliases
	using DSLBinding = VkDescriptorSetLayoutBinding;
	using DSLBindings = std::vector <DSLBinding>;

	// All of the layer's cameras
	std::vector <Camera>		_cameras;

	// Active camera
	Camera				*_active_camera = nullptr;

	// Application context
	App::Window			_wctx;

	// Render pass
	VkRenderPass			_render_pass;

	// All rendering pipelines
	struct {
		Vulkan::Pipeline	albedo;
		Vulkan::Pipeline	normals;
		Vulkan::Pipeline	blinn_phong;
	} _pipelines;

	// Descriptor set and layout
	VkDescriptorSetLayout		_full_dsl;

	// Buffers for lights
	BufferManager <uint8_t>		_ubo_point_lights_buffer;

	// Refresh a buffer with its data
	static void _refresh(BufferManager <uint8_t> &buffer, const uint8_t *data, size_t size,
			const VkDescriptorSet &ds, size_t binding) {
		buffer.write(data, size);
		buffer.sync_upload();
		buffer.bind(ds, binding);
	}

	// Corresponding structures
	UBO_PointLights			_ubo_point_lights;

	// Current rendering mode
	Mode				_mode = Mode::ALBEDO;

	// Get current pipeline
	Vulkan::Pipeline *_get_pipeline() {
		switch (_mode) {
		case Mode::ALBEDO:
			return &_pipelines.albedo;
		case Mode::NORMAL:
			return &_pipelines.normals;
		case Mode::BLINN_PHONG:
			return &_pipelines.blinn_phong;
		default:
			break;
		}

		return nullptr;
	}

	// Initialize Vulkan structures
	void _initialize_vulkan_structures(const VkAttachmentLoadOp &);

	// Descriptor set bindings
	static const DSLBindings	_full_dsl_bindings;

	// Highlight status and methods
	std::vector <bool>		_highlighted;
public:
	// Default
	Layer() = default;

	// Constructor
	// TODO: inherit, and add a extra variable to check initialization
	// status
	Layer(const App::Window &wctx, const VkAttachmentLoadOp &load = VK_ATTACHMENT_LOAD_OP_LOAD)
			: _wctx(wctx) {
		_initialize_vulkan_structures(load);
	}

	// Adding elements
	void add_do(const ptr &e) override {
		// Prepare latching packet
		LatchingPacket lp {
			.context = &_wctx.context,
			.command_pool = &_wctx.command_pool,
			.ubo_point_lights = &_ubo_point_lights,
			.layer = this
		};

		// Add element
		e->latch(lp);

		// Add highlight element
		_highlighted.push_back(false);
		
		// Refresh lights for all elements
		for (int i = 0; i < _elements.size(); i++) {
			// Update once for all element, bind once for all
			_refresh(_ubo_point_lights_buffer,
				(const uint8_t *) &_ubo_point_lights,
				sizeof(_ubo_point_lights),

				_elements[i]->get_local_ds(),
				RASTER_BINDING_POINT_LIGHTS
			);
		}
	}

	// Import scene objects
	void add_scene(const Scene &) override;

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

	// Set active camera
	void set_active_camera(const Camera &camera) {
		// If active camera has not been set
		if (_active_camera == nullptr) {
			if (_cameras.empty())
				_cameras.push_back(camera);

			_active_camera = &_cameras[0];
		}

		*_active_camera = camera;
	}

	// Set rendering mode
	void set_mode(const Mode &mode) {
		_mode = mode;
	}

	// Set highlight
	void set_highlight(size_t index, bool highlight) {
		if (index < _highlighted.size()) {
			_highlighted[index] = highlight;
		} else {
			KOBRA_LOG_FUNC(warn) << "Highlight index out of range ["
				<< index << "/" << _highlighted.size() << "]";
		}
	}

	// Clear highlighting
	void clear_highlight() {
		_highlighted = std::vector <bool> (_highlighted.size(), false);
	}

	// Serve a descriptor set
	VkDescriptorSet serve_ds() {
		return _wctx.context.make_ds(
			_wctx.descriptor_pool,
			_full_dsl
		);
	}

	// Render
	void render(const VkCommandBuffer &cmd_buffer, const VkFramebuffer &framebuffer) {
		// Check initialization status
		if (_get_pipeline()->pipeline == VK_NULL_HANDLE) {
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
			_get_pipeline()->pipeline
		);

		/* Bind descriptor set
		vkCmdBindDescriptorSets(cmd_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_get_pipeline()->layout,
			0, 1, &_full_ds,
			0, nullptr
		); */

		// Initialize render packet
		RenderPacket packet {
			.cmd = cmd_buffer,

			.pipeline_layout = _get_pipeline()->layout,

			// TODO: warn on null camera
			.view = _active_camera->view(),
			.proj = _active_camera->perspective()
		};

		// Render all elements
		for (int i = 0; i < _elements.size(); i++) {
			packet.highlight = _highlighted[i];
			/* _refresh(_ubo_point_lights_buffer,
				(const uint8_t *) &_ubo_point_lights,
				sizeof(_ubo_point_lights),

				_elements[i]->get_local_ds(),
				RASTER_BINDING_POINT_LIGHTS
			); */

			_elements[i]->render(packet);
		}

		// End render pass
		vkCmdEndRenderPass(cmd_buffer);
	}
};

}

}

#endif
