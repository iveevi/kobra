#ifndef KOBRA_RT_LAYER_H_
#define KOBRA_RT_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
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

	// Pipelines
	struct {
		Vulkan::Pipeline _mesh_pipeline;
	} _pipelines;

	// Descriptor set bindings
	static const DSLBindings _mesh_compute_bindings;

	VkDescriptorSetLayout	_mesh_dsl;

	// Initialize pipelines
	void _init_pipelines() {
		// First, create the DSL

		// Mesh pipeline

		// TODO: context method to create layout
		VkPipelineLayoutCreateInfo mesh_ppl_ci = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			// .pSetLayouts = &_context.get_descriptor_set_layout(),
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};
	}
public:
	// Default constructor
	Layer() = default;

	// Constructor
	Layer(const App::Window &wctx)
			: _context(wctx.context) {
		// Initialize pipelines
		_init_pipelines();
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
};

}

}

#endif
