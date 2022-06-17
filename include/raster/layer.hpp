#ifndef KOBRA_RASTER_LAYER_H_
#define KOBRA_RASTER_LAYER_H_

// Standard headers
#include <vector>

// Vulkan headers
#include <vulkan/vulkan_core.h>

// Engine headers
#include "../../shaders/raster/bindings.h"
#include "../app.hpp"
// #include "../buffer_manager.hpp"
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
	// All of the layer's cameras
	std::vector <Camera>		_cameras;

	// Active camera
	Camera				*_active_camera = nullptr;

	// Vulkan structures
	const vk::raii::PhysicalDevice	&_physical_device = nullptr;
	const vk::raii::Device		&_device = nullptr;
	const vk::raii::CommandPool	&_command_pool = nullptr;
	const vk::raii::DescriptorPool	&_descriptor_pool = nullptr;

	vk::raii::RenderPass		_render_pass = nullptr;

	// Other layer properties
	vk::Extent2D			_extent;

	// All rendering pipelines
	struct {
		// Pipelines
		vk::raii::Pipeline		albedo = nullptr;
		vk::raii::Pipeline		normals = nullptr;
		vk::raii::Pipeline		blinn_phong = nullptr;

		// Common pipeline layout
		vk::raii::PipelineLayout	layout = nullptr;
	} _pipelines;

	// Descriptor set and layout
	vk::raii::DescriptorSetLayout	_dsl_full = nullptr;

	// Buffers for lights
	BufferData			_ubo_point_lights_buffer = nullptr;

	/* Refresh a buffer with its data
	void _refresh(BufferData &buffer,
			const uint8_t *data,
			size_t size,
			const vk::raii::DescriptorSet &dset,
			size_t binding) {
		buffer.upload(data, size);
		bind_ds(_device, dset, buffer, vk::DescriptorType::eUniformBuffer, binding);
	} */

	// Corresponding structures
	UBO_PointLights			_ubo_point_lights;

	// Current rendering mode
	Mode				_mode = Mode::ALBEDO;

	// Initialization status
	bool				_initialized = false;

	// Get current pipeline
	const vk::raii::Pipeline &_get_pipeline() {
		switch (_mode) {
		case Mode::ALBEDO:
			return _pipelines.albedo;
		case Mode::NORMAL:
			return _pipelines.normals;
		case Mode::BLINN_PHONG:
			return _pipelines.blinn_phong;
		default:
			break;
		}

		throw std::runtime_error("Invalid rendering mode");
		return _pipelines.albedo;
	}

	// Initialize Vulkan structures
	void _initialize_vulkan_structures(const vk::AttachmentLoadOp &,
			const vk::Format &, const vk::Format &);

	// Descriptor set bindings
	static const std::vector <DSLB>	_full_dsl_bindings;
public:
	// Default
	Layer() = default;

	// Constructor
	// TODO: inherit, and add a extra variable to check initialization
	// status
	Layer(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const vk::raii::DescriptorPool &descriptor_pool,
			const vk::Extent2D &extent,
			const vk::Format &swapchain_format,
			const vk::Format &depth_format,
			const vk::AttachmentLoadOp &load = vk::AttachmentLoadOp::eLoad)
			: _physical_device(phdev),
			_device(device),
			_command_pool(command_pool),
			_descriptor_pool(descriptor_pool),
			_extent(extent) {
		_initialize_vulkan_structures(load, swapchain_format, depth_format);
		_initialized = true;
	}

	// Move constructor and assignment operator
	// Layer(Layer &&other) noexcept = default;
	// Layer &operator=(Layer &&other) noexcept = default;

	// Adding elements
	void add_do(const ptr &e) override {
		// Prepare latching packet
		LatchingPacket lp {
			.phdev = _physical_device,
			.device = _device,
			.command_pool = _command_pool,
			.layer = this
		};

		// Add element
		e->latch(lp);

		// Bind descriptor
		bind_ds(_device, e->get_local_ds(),
			_ubo_point_lights_buffer,
			vk::DescriptorType::eUniformBuffer,
			RASTER_BINDING_POINT_LIGHTS
		);
	}

	// Import scene objects
	void add_scene(const Scene &) override;

	// Erase by name
	void erase(const std::string &name) {
		for (auto itr = _elements.begin(); itr != _elements.end(); ++itr) {
			if ((*itr)->name() == name) {
				_elements.erase(itr);
				return;
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

	// Index an element
	ptr operator[](size_t index) {
		return _elements[index];
	}

	// Index an element by name
	ptr operator[](const std::string &name) {
		for (auto &e : _elements) {
			if (e->name() == name)
				return e;
		}

		return nullptr;
	}

	// Clear highlighting
	void clear_highlight() {
		for (auto &e : _elements)
			e->highlight = false;
	}

	// Serve a descriptor set
	vk::raii::DescriptorSet serve_ds() {
		auto dsets = vk::raii::DescriptorSets {
			_device,
			{*_descriptor_pool, *_dsl_full}
		};

		return std::move(dsets.front());
	}

	// Render
	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer,
			const vk::Extent2D &extent) {
		// Check initialization status
		if (_initialized == false) {
			KOBRA_LOG_FUNC(warn) << "calling ::render() on"
				" raster::Layer (" << this << ") which has not"
				" been yet been initialized\n";
		}

		// Set viewport
		auto viewport = vk::Viewport {
			0.0f, 0.0f,
			static_cast<float>(_extent.width),
			static_cast<float>(_extent.height),
			0.0f, 1.0f
		};

		cmd.setViewport(0, viewport);

		// Set scissor
		auto scissor = vk::Rect2D {
			vk::Offset2D {0, 0},
			_extent
		};

		cmd.setScissor(0, scissor);

		// First update the lighting
		_ubo_point_lights.number = 0;
		LightingPacket lp {
			.ubo_point_lights = &_ubo_point_lights,
		};

		for (auto &e : _elements)
			e->light(lp);

		// Upload to GPU
		_ubo_point_lights_buffer.upload(
			(const uint8_t *) &_ubo_point_lights,
			sizeof(_ubo_point_lights),
			false
		);

		// Start render pass (clear both color and depth)
		std::array <vk::ClearValue, 2> clear_values = {
			vk::ClearValue {
				vk::ClearColorValue {
					std::array <float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
				}
			},
			vk::ClearValue {
				vk::ClearDepthStencilValue {
					1.0f, 0
				}
			}
		};

		cmd.beginRenderPass(
			vk::RenderPassBeginInfo {
				*_render_pass,
				*framebuffer,
				vk::Rect2D {
					vk::Offset2D {0, 0},
					extent,
				},
				static_cast <uint32_t> (clear_values.size()),
				clear_values.data()
			},
			vk::SubpassContents::eInline
		);

		// Bind pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*_get_pipeline()
		);

		/* Refresh for all elements
		for (auto &e : _elements) {
			_refresh(
				_ubo_point_lights_buffer,
				(const uint8_t *) &_ubo_point_lights,
				sizeof(_ubo_point_lights),
				e->get_local_ds(),
				RASTER_BINDING_POINT_LIGHTS
			);
		} */

		// Initialize render packet
		RenderPacket packet {
			.cmd = cmd,
			.pipeline_layout = _pipelines.layout,

			// TODO: warn on null camera
			.view = _active_camera->view(),
			.proj = _active_camera->perspective()
		};

		// Render all elements
		for (int i = 0; i < _elements.size(); i++) {
			// packet.highlight = _highlighted[i];
			// TODO: if this approach works, then remove this
			// variable from the struct
			packet.highlight = _elements[i]->highlight;
			_elements[i]->render(packet);
		}

		// End render pass
		cmd.endRenderPass();
	}
};

}

}

#endif
