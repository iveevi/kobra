#ifndef RASTER_LAYER_H_
#define RASTER_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "../app.hpp"
#include "../camera.hpp"
#include "../scene.hpp"
#include "pipeline.hpp"
#include "raster.hpp"

namespace kobra {

namespace raster {

// Layer class holds
//	all the elements that need
//	to be rendered
class Layer {
public:
	// Rendering mode
	enum class Mode {
		ALBEDO,
		NORMAL,
		BLINN_PHONG
	};
protected:
	// All of the layer's cameras
	std::vector <Camera>	_cameras;

	// Active camera
	Camera			*_active_camera = nullptr;

	// List of elements
	// TODO: map elements to their respective pipelines
	std::vector <Element>	_elements;

	// Application context
	App::Window		_wctx;

	// Render pass
	VkRenderPass		_render_pass;

	// All rendering pipelines
	struct {
		Vulkan::Pipeline	albedo;
		Vulkan::Pipeline	normals;
		Vulkan::Pipeline	blinn_phong;
	} _pipelines;

	// Current rendering mode
	Mode			_mode = Mode::ALBEDO;

	// Get current pipeline
	Vulkan::Pipeline *get_pipeline() {
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
		// TODO: create a map of names to shaders (for fragment, since
		// vertex is the same)
		std::vector <VkShaderModule> shaders = _wctx.context.make_shaders({
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/color_frag.spv",
			"shaders/bin/raster/normal_frag.spv",
			"shaders/bin/raster/blinn_phong_frag.spv"
		});

		// Create pipelines
		// TODO: move this make_pipline function here
		_pipelines.albedo = make_pipeline(
			_wctx,
			_render_pass,
			shaders[0],
			shaders[1]
		);

		_pipelines.normals = make_pipeline(
			_wctx,
			_render_pass,
			shaders[0],
			shaders[2]
		);

		_pipelines.blinn_phong = make_pipeline(
			_wctx,
			_render_pass,
			shaders[0],
			shaders[3]
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
	Layer(const App::Window &wctx, const VkAttachmentLoadOp &load = VK_ATTACHMENT_LOAD_OP_LOAD)
			: _wctx(wctx) {
		_initialize_vulkan_structures(load);
		_initialized = true;
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

	// Import scene objects
	void add(const Scene &scene) {
		for (const ObjectPtr &obj : scene) {
			// TODO: latre also add cameras
			if (obj->type() == raster::Mesh::object_type) {
				raster::Mesh *mesh = dynamic_cast
					<raster::Mesh *> (obj.get());
				_elements.push_back(Element(mesh));
			} else if (obj->type() == kobra::Mesh::object_type) {
				kobra::Mesh *mesh = dynamic_cast
					<kobra::Mesh *> (obj.get());
				raster::Mesh *raster_mesh = new raster::Mesh(_wctx.context, *mesh);
				_elements.push_back(Element(raster_mesh));
			}
		}
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
			get_pipeline()->pipeline
		);

		// Initialize render packet
		RenderPacket packet {
			.cmd = cmd_buffer,

			.pipeline_layout = get_pipeline()->layout,

			// TODO: warn on null camera
			.view = _active_camera->view(),
			.proj = _active_camera->perspective()
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
