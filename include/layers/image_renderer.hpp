#ifndef KOBRA_LAYERS_IMAGE_RENDERER_H_
#define KOBRA_LAYERS_IMAGE_RENDERER_H_

// Engine headers
#include "common.hpp"
#include "../backend.hpp"

namespace kobra {

namespace layers {

class ImageRenderer {
	// Critical Vulkan structures
	vk::raii::Device *m_device = nullptr;
	vk::raii::PhysicalDevice *m_phdev = nullptr;
	vk::raii::DescriptorPool *m_descriptor_pool = nullptr;

	// Vulkan structures
	vk::raii::RenderPass m_render_pass = nullptr;
	vk::raii::Pipeline m_pipeline = nullptr;
	vk::raii::PipelineLayout m_ppl = nullptr;
	vk::raii::DescriptorSetLayout m_dsl = nullptr;
	
	// Synchronization queue for image updates
	SyncQueue *m_sync_queue;

	// Rendering resources
	BufferData m_buffer_data = nullptr;
	vk::ImageView m_prev_image_view = nullptr;
	vk::raii::DescriptorSet m_dset = nullptr;
	vk::raii::Sampler m_sampler = nullptr;
public:
	// Constructors
	ImageRenderer() = default;
	ImageRenderer(const Context &);

	void render(const ImageData &, const RenderContext &);
};

}

}

#endif
