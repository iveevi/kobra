#include "../../include/ecs.hpp"
#include "../../include/layers/objectifier.hpp"
#include "../../include/vertex.hpp"

namespace kobra {

namespace layers {

// Push constants
struct PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	uint32_t id;
};

// Create the layer
Objectifier make_layer(const Context &context)
{
	static vk::Format format = vk::Format::eR32Uint;

	// To return
	Objectifier layer;

	// Create the image
	layer.image = ImageData(*context.phdev, *context.device,
		format, context.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment
			| vk::ImageUsageFlagBits::eTransferSrc,
		vk::ImageLayout::eUndefined,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	// Create the staging buffer
	vk::DeviceSize size = context.extent.width
		* context.extent.height * sizeof(uint32_t);

	layer.staging_buffer = BufferData(*context.phdev, *context.device,
		size,
		vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Create the depth buffer
	layer.depth_buffer = DepthBuffer {
		*context.phdev, *context.device,
		vk::Format::eD32Sfloat,
		context.extent
	};

	// Create the render pass
	layer.render_pass = make_render_pass(*context.device,
		{format},
		{vk::AttachmentLoadOp::eClear},
		layer.depth_buffer.format,
		vk::AttachmentLoadOp::eClear
	);

	// Create the framebuffer
	vk::ImageView attachments[] = {
		*layer.image.view,
		*layer.depth_buffer.view
	};

	vk::FramebufferCreateInfo framebuffer_info {
		{}, *layer.render_pass,
		2, &attachments[0],
		context.extent.width, context.extent.height, 1
	};

	layer.framebuffer = vk::raii::Framebuffer(*context.device,
		framebuffer_info
	);

	// Push constants and pipeline layout
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	layer.ppl = vk::raii::PipelineLayout {
		*context.device,
		{{}, {}, push_constants}
	};

	// Create the pipeline
	auto shaders = make_shader_modules(*context.device, {
		"bin/spv/objectifier_vert.spv",
		"bin/spv/objectifier_frag.spv"
	});

	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo grp_info {
		*context.device, layer.render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		vertex_binding, vertex_attributes,
		layer.ppl
	};

	grp_info.blend_enabled = false;

	layer.pipeline = make_graphics_pipeline(grp_info);

	// Return
	return layer;
}

// Render entities and download the image
void render(Objectifier &layer,
		const CommandBuffer &cmd,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{
	// Default render area (viewport and scissor)
	RenderArea ra {{-1, -1}, {-1, -1}};
	ra.apply(cmd, layer.image.extent);

	// Clear colors
	// TODO: easier function to use
	std::array <vk::ClearValue, 2> clear_values {
		vk::ClearColorValue {
			std::array <float, 4> {0, 0, 0, 0}
		},

		vk::ClearValue {
			vk::ClearDepthStencilValue {
				1.0f, 0
			}
		}
	};

	// Begin render pass
	cmd.beginRenderPass(
		vk::RenderPassBeginInfo {
			*layer.render_pass,
			*layer.framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				layer.image.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Bind the pipeline
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *layer.pipeline);

	// Setup push constants
	PushConstants pc {
		.view = camera.view_matrix(transform),
		.projection = camera.perspective_matrix()
	};

	// Render all entities (with rasterizers)
	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <Rasterizer> (i)) {
			pc.model = ecs.get <Transform> (i).matrix();
			pc.id = i;

			cmd.pushConstants <PushConstants> (*layer.ppl,
				vk::ShaderStageFlagBits::eVertex,
				0, pc
			);

			// Bind and draw
			const Rasterizer &rasterizer = ecs.get <Rasterizer> (i);

			int submeshes = rasterizer.size();
			for (int i = 0; i < submeshes; i++) {
				cmd.bindVertexBuffers(0, *rasterizer.get_vertex_buffer(i).buffer, {0});
				cmd.bindIndexBuffer(*rasterizer.get_index_buffer(i).buffer,
					0, vk::IndexType::eUint32
				);

				cmd.drawIndexed(rasterizer.get_index_count(i), 1, 0, 0, 0);
			}
		}
	}

	// End render pass
	cmd.endRenderPass();

	// Download the image
	layer.image.transition_layout(cmd,
	     vk::ImageLayout::eTransferSrcOptimal
	);

	copy_image_to_buffer(cmd,
		layer.image.image, layer.staging_buffer.buffer,
		layer.image.format, layer.image.extent.width,
		layer.image.extent.height
	);

	// Wait for the copy to finish
	cmd.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eHost,
		{},
		nullptr, nullptr, nullptr
	);

	// Wait for the download to finish
	cmd.pipelineBarrier(
		vk::PipelineStageFlagBits::eHost,
		vk::PipelineStageFlagBits::eTransfer,
		{},
		nullptr, nullptr, nullptr
	);

	// Transition back to color attachment
	layer.image.transition_layout(cmd,
		vk::ImageLayout::ePresentSrcKHR
	);
}

}

}
