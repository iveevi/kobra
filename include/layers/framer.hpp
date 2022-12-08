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
	// Critical Vulkan structures
	vk::raii::Device *m_device = nullptr;
	vk::raii::PhysicalDevice *m_phdev = nullptr;
	vk::raii::DescriptorPool *m_descriptor_pool = nullptr;

	// Local Vulkan structures
	vk::raii::RenderPass m_render_pass = nullptr;

	vk::raii::Pipeline m_pipeline = nullptr;
	vk::raii::PipelineLayout m_ppl = nullptr;

	// Descriptor set layout
	vk::raii::DescriptorSetLayout m_dsl = nullptr;

	// Descriptor sets
	vk::raii::DescriptorSet m_dset = nullptr;

	// Synchronization handle
	SyncQueue *m_sync_queue = nullptr;

	// Data for rendering
	ImageData m_result_image = nullptr;
	BufferData m_result_buffer = nullptr;
	vk::raii::Sampler m_result_sampler = nullptr;

	// Private helper functions
	void resize_callback(const Image &);

	// Static member variables
	static const std::vector <DescriptorSetLayoutBinding>
		DESCRIPTOR_SET_BINDINGS;
public:
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
