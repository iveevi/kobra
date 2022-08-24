#ifndef KOBRA_LAYERS_RASTERIZER_H_
#define KOBRA_LAYERS_RASTERIZER_H_

// Standard headers
#include <map>

// Engine headers
// TODO: move layer.hpp to this directory
// #include "../layer.hpp"

#include "../backend.hpp"
#include "../ecs.hpp"
#include "../../shaders/raster/bindings.h"
#include "../../shaders/raster/constants.h"
#include "../vertex.hpp"

namespace kobra {

namespace layers {

class Raster {
private:
	// Vulkan context
	Context				_ctx;

	// Other Vulkan structures
	vk::raii::RenderPass		_render_pass = nullptr;

	vk::raii::PipelineLayout	_ppl = nullptr;
	vk::raii::Pipeline		_p_albedo = nullptr;
	vk::raii::Pipeline		_p_normal = nullptr;
	vk::raii::Pipeline		_p_phong = nullptr;

	// Buffer for all the lights
	struct LightsData;

	BufferData			_b_lights = nullptr;

	// Bind pipeline from raster mode
	const vk::raii::Pipeline &get_pipeline(RasterMode);

	// Descriptor set layout and bindings
	vk::raii::DescriptorSetLayout	_dsl = nullptr;

	static const std::vector <DSLB>	_dsl_bindings;

	// Create a descriptor set
	vk::raii::DescriptorSet	_make_ds() const;

	// Rasterizer components to descriptor set
	Rasterizer::ResourceMap		_ds_components;

	// Box mesh for area lights
	Rasterizer			*_area_light;
public:
	// Default constructor
	Raster() = default;

	// Constructors
	Raster(const Context &, const vk::AttachmentLoadOp &);

	// Render
	void render(const vk::raii::CommandBuffer &,
			const vk::raii::Framebuffer &,
			const ECS &,
			const RenderArea & = RenderArea {{-1, -1}, {-1, -1}});
};

}

}

#endif
