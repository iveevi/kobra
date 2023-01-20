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
class Renderable;

namespace layers {

struct ForwardRenderer {
	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;

	// Vulkan structures
	vk::raii::RenderPass render_pass = nullptr;

	// TODO: map of pipelines, for each rasterizer...
	vk::raii::Pipeline pipeline = nullptr;

	// Pipeline layout is shared across all fragment shaders
	vk::raii::PipelineLayout ppl = nullptr;

	// Texture loader
	TextureLoader *loader = nullptr;

	vk::Extent2D extent = { 0, 0 };

	// Descriptor set bindings
	static const std::vector <DSLB> dsl_bindings;

	// Descriptor set layout
	vk::raii::DescriptorSetLayout dsl = nullptr;

	// Descriptor sets
	using RenderableDset = std::vector <vk::raii::DescriptorSet>;

	std::map <const Renderable *, RenderableDset> dsets;

	// Constructors
	ForwardRenderer() = default;
	ForwardRenderer(const Context &);

	// TODO: add extra and default shader pograms here...
	// TODO: parameters into a struct
	void render(const ECS &,
		const Camera &,
		const Transform &,
		const vk::raii::CommandBuffer &,
		const vk::raii::Framebuffer &,
		const RenderArea & = RenderArea::full());
};

}

}

#endif
