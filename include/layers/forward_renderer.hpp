#ifndef KOBRA_LAYERS_FORWARD_RENDERER_H_
#define KOBRA_LAYERS_FORWARD_RENDERER_H_

// Standard headers
#include <map>
#include <vector>

// Engine headers
#include "../backend.hpp"
#include "../vertex.hpp"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;
class Rasterizer;

namespace layers {

struct ForwardRenderer {
	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;
	
	// Vulkan structures
	RenderPass render_pass = nullptr;

	// TODO: map of pipelines, for each rasterizer...
	Pipeline pipeline = nullptr;

	// Pipeline layout is shared across all fragment shaders
	PipelineLayout ppl = nullptr;

	vk::Extent2D extent = { 0, 0 };

	// Descriptor set bindings
	static const std::vector <DSLB> dsl_bindings;

	// Descriptor set layout
	vk::raii::DescriptorSetLayout dsl = nullptr;

	// Descriptor sets
	using RasterizerDset = std::vector <vk::raii::DescriptorSet>;

	std::map <const Rasterizer *, RasterizerDset> dsets;

	// TODO: add extra and default shader pograms here...

	// Functions
	static ForwardRenderer make(const Context &);
};

// Other methods
// TODO: parameters into a struct
void render(ForwardRenderer &,
	const ECS &,
	const Camera &,
	const Transform &,
	const CommandBuffer &,
	const Framebuffer &,
	const RenderArea & = {{-1, -1}, {-1, -1}});

}

}

#endif
