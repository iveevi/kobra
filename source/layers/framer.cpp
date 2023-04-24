#include "../../include/layers/framer.hpp"

namespace kobra {

namespace layers {

// Static member variables
const std::vector <DescriptorSetLayoutBinding> Framer::DESCRIPTOR_SET_BINDINGS = {
	DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

// Create the layer
Framer::Framer(const Context &context, const vk::raii::RenderPass &render_pass)
		: m_device(context.device),
		m_phdev(context.phdev),
		m_sync_queue(context.sync_queue)
{
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
		*context.device, render_pass,
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
        // TODO: upgrade to 32 bit float
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
void Framer::pre_render(const vk::raii::CommandBuffer &cmd, const RawImage &frame)
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
}	

void Framer::render(const vk::raii::CommandBuffer &cmd)
{
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
}

}

}
