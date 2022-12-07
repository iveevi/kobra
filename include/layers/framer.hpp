#ifndef KOBRA_LAYERS_FRAMER_H_
#define KOBRA_LAYERS_FRAMER_H_

// Engine headers
#include "../backend.hpp"
#include "../image.hpp"

namespace kobra {

namespace layers {

// The purpose of this layer is to render an image frame onto the screen; the
// said image frame is a simple array of pixels of particular format
// TODO: allow formats other than RGBA
class Framer {
	void resize_callback(const Image &);
public:
	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;

	// Local Vulkan structures
	vk::raii::RenderPass render_pass = nullptr;

	vk::raii::Pipeline pipeline = nullptr;
	vk::raii::PipelineLayout ppl = nullptr;

	// Descriptor set bindings
	static const std::vector <DSLB> dsl_bindings;

	// Descriptor set layout
	vk::raii::DescriptorSetLayout dsl = nullptr;

	// Descriptor sets
	vk::raii::DescriptorSet dset = nullptr;

	// Synchronization handle
	SyncQueue *sync_queue = nullptr;

	// Data for rendering
	ImageData result_image = nullptr;
	BufferData result_buffer = nullptr;
	vk::raii::Sampler result_sampler = nullptr;

	// Default constructor
	Framer() = default;

	// Constructor
	Framer(const Context &);

	// Render image frame to screen
	// TODO: pack command buffer, frame buffer and render area into a struct
	// (layer struct...) and use as the first argument
	void render(const Image &,
		const vk::raii::CommandBuffer &,
		const vk::raii::Framebuffer &,
		const vk::Extent2D &,
		const RenderArea & = {{-1, -1}, {-1, -1}});
};

}

}

#endif
