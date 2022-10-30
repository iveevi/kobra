#include "../../include/layers/framer.hpp"

namespace kobra {

namespace layers {

// Static member variables
const std::vector <DSLB> Framer::dsl_bindings = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

// Create the layer
// TODO: all custom extent...
Framer Framer::make(const Context &context)
{
	// To return
	Framer layer;

	// Extract critical Vulkan structures
	layer.device = context.device;
	layer.phdev = context.phdev;
	layer.descriptor_pool = context.descriptor_pool;
	layer.extent = context.extent;

	// Create the present render pass
	layer.render_pass = make_render_pass(*context.device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Descriptor set layout
	layer.dsl = make_descriptor_set_layout(*context.device, dsl_bindings);

	// Allocate present descriptor set
	auto dsets = vk::raii::DescriptorSets {
		*context.device,
		{**context.descriptor_pool, *layer.dsl}
	};

	layer.dset = std::move(dsets.front());

	// Push constants and pipeline layout
	layer.ppl = PipelineLayout {
		*context.device,
		{{}, *layer.dsl, {}}
	};

	// Create the present pipeline
	auto shaders = make_shader_modules(*context.device, {
		"bin/spv/spit_vert.spv",
		"bin/spv/spit_frag.spv"
	});
	
	GraphicsPipelineInfo present_grp_info {
		*context.device, layer.render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		{}, {},
		layer.ppl
	};

	present_grp_info.no_bindings = true;
	present_grp_info.depth_test = false;
	present_grp_info.depth_write = false;

	layer.pipeline = make_graphics_pipeline(present_grp_info);

	// Allocate resources for rendering results

	// TODO: shared resource as a CUDA texture?
	layer.result_image = ImageData(
		*context.phdev, *context.device,
		vk::Format::eR8G8B8A8Unorm,
		context.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		vk::ImageLayout::eUndefined,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	layer.result_sampler = make_sampler(*context.device, layer.result_image);

	// Allocate staging buffer
	vk::DeviceSize stage_size = context.extent.width
		* context.extent.height
		* sizeof(uint32_t);

	auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
	auto mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal
		| vk::MemoryPropertyFlagBits::eHostCoherent
		| vk::MemoryPropertyFlagBits::eHostVisible;

	layer.result_buffer = BufferData(
		*context.phdev, *context.device, stage_size,
		usage | vk::BufferUsageFlagBits::eTransferSrc, mem_props
	);

	// Bind image sampler to the present descriptor set
	//	immediately, since it will not change
	bind_ds(*context.device,
		layer.dset,
		layer.result_sampler,
		layer.result_image, 0
	);

	// Return
	return layer;
}

// Render to the presentable framebuffer
// TODO: custom extent
void render(Framer &layer,
		const std::vector <uint32_t> &frame,
		const CommandBuffer &cmd,
		const Framebuffer &framebuffer,
		const RenderArea &ra)
{
	// Upload data to the buffer
	layer.result_buffer.upload(frame);
	
	// Copy buffer to image
	layer.result_image.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

	copy_data_to_image(cmd,
		layer.result_buffer.buffer,
		layer.result_image.image,
		layer.result_image.format,
		layer.extent.width, layer.extent.height
	);

	// Transition image back to shader read
	layer.result_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);	
	
	// Apply render area
	ra.apply(cmd, layer.extent);

	// Clear colors
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
			*layer.render_pass,
			*framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				layer.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Presentation pipeline
	cmd.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		*layer.pipeline
	);

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*layer.ppl, 0, {*layer.dset}, {}
	);

	// Draw and end
	cmd.draw(6, 1, 0, 0);
	cmd.endRenderPass();
}

}

}
