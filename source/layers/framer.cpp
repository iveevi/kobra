#include "../../include/layers/framer.hpp"

namespace kobra {

namespace layers {

// Static member variables
const std::vector <DescriptorSetLayoutBinding>
		Framer::DESCRIPTOR_SET_BINDINGS = {
	DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

// Create the layer
Framer::Framer(const Context &context)
		: m_device(context.device),
		m_phdev(context.phdev),
		m_descriptor_pool(context.descriptor_pool),
		m_sync_queue(context.sync_queue)
{
	// Create the present render pass
	m_render_pass = make_render_pass(*context.device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Descriptor set layout
	m_dsl = make_descriptor_set_layout(
		*context.device,
		DESCRIPTOR_SET_BINDINGS
	);

	// Allocate present descriptor set
	auto dsets = vk::raii::DescriptorSets {
		*context.device,
		{**context.descriptor_pool, *m_dsl}
	};

	m_dset = std::move(dsets.front());

	// Push constants and pipeline layout
	m_ppl = vk::raii::PipelineLayout {
		*context.device,
		{{}, *m_dsl, {}}
	};

	// Create the present pipeline
	auto shaders = make_shader_modules(*context.device, {
		KOBRA_SHADERS_DIR "/spit_vert.spv",
		KOBRA_SHADERS_DIR "/spit_frag.spv"
	});
	
	GraphicsPipelineInfo present_grp_info {
		*context.device, m_render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		{}, {},
		m_ppl
	};

	present_grp_info.no_bindings = true;
	present_grp_info.depth_test = false;
	present_grp_info.depth_write = false;

	m_pipeline = make_graphics_pipeline(present_grp_info);

	// Allocate resources for rendering results
	m_result_image = ImageData(
		*context.phdev, *context.device,
		vk::Format::eR8G8B8A8Unorm,
		context.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		// vk::ImageLayout::eUndefined,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	m_result_sampler = make_continuous_sampler(*context.device);

	// Allocate staging buffer
	vk::DeviceSize stage_size = context.extent.width
		* context.extent.height
		* sizeof(uint32_t);

	auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
	auto mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal
		| vk::MemoryPropertyFlagBits::eHostCoherent
		| vk::MemoryPropertyFlagBits::eHostVisible;

	m_result_buffer = BufferData(
		*context.phdev, *context.device, stage_size,
		usage | vk::BufferUsageFlagBits::eTransferSrc, mem_props
	);

	// Bind image sampler to the present descriptor set
	//	immediately, since it will not change
	bind_ds(*context.device,
		m_dset,
		m_result_sampler,
		m_result_image, 0
	);
}

// Resize callback
void Framer::resize_callback(const RawImage &frame)
{
	// Resize resources
	m_result_buffer.resize(frame.size());

	m_result_image = ImageData(
		*m_phdev, *m_device,
		vk::Format::eR8G8B8A8Unorm,
		{frame.width, frame.height},
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		// vk::ImageLayout::eUndefined,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	// m_result_sampler = make_sampler(*m_device, m_result_image);

	bind_ds(*m_device,
		m_dset,
		m_result_sampler,
		m_result_image, 0
	);
}

// Render to the presentable framebuffer
void Framer::render
		(const RawImage &frame,
		const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const vk::Extent2D &extent,
		const RenderArea &ra)
{
	if (!m_sync_queue) {
		// TODO: remove the default constructor, force pointer
		KOBRA_LOG_FUNC(Log::ERROR) << "Framer: null sync queue, was the framer initialized?\n";
		throw std::runtime_error("Framer: null sync queue");
	}

	// Upload data to the buffer
	// TODO: also allow resize... pass an image struct instead
	bool skip_frame = false;
	if (m_result_buffer.size != frame.size()) {
		// Sync changes
		m_sync_queue->push({
			"[Framer] Resized resources",
			[&, frame] () {
				resize_callback(frame);
			}
		});

		skip_frame = true;
	}

	if (!skip_frame) {
		m_result_buffer.upload(frame.data);
		
		// Copy buffer to image
		m_result_image.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

		copy_data_to_image(cmd,
			m_result_buffer.buffer,
			m_result_image.image,
			m_result_image.format,
			frame.width, frame.height
		);
	}

	// Transition image back to shader read
	m_result_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
		
	// Apply render area
	ra.apply(cmd, extent);

	// Clear colors
	// TODO: method
	std::array <vk::ClearValue, 2> clear_values {
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
	
	// Start the render pass
	cmd.beginRenderPass(
		vk::RenderPassBeginInfo {
			*m_render_pass,
			*framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Presentation pipeline
	cmd.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		*m_pipeline
	);

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*m_ppl, 0, {*m_dset}, {}
	);

	// Draw and end
	cmd.draw(6, 1, 0, 0);
	cmd.endRenderPass();
}

}

}
