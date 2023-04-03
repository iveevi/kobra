#include "../../include/ecs.hpp"
#include "../../include/layers/objectifier.hpp"
#include "../../include/vertex.hpp"
#include "../../include/shader_program.hpp"
#include <vulkan/vulkan_enums.hpp>

namespace kobra {

namespace layers {

// Push constants
struct PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	glm::uvec2 id;
};

struct CompositingData {
	glm::uvec2 expected;
};

// TODO: for lights, render a quad encompassing the texture

// Create the layer
Objectifier::Objectifier(const Context &context)
{
	// TODO: RG (mesh, submesh)
	static vk::Format format = vk::Format::eR32G32Uint;

	// Load all shader programs
	std::string vertex_shader_source = common::read_file(KOBRA_DIR "/source/shaders/objectifier.vert");
	std::string rendering_fragment_shader_source = common::read_file(KOBRA_DIR "/source/shaders/objectifier_rendering.frag");
	std::string compositing_fragment_shader_source = common::read_file(KOBRA_DIR "/source/shaders/objectifier_compositing.frag");

	auto vertex_shader_program = ShaderProgram(
		vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	);

	auto opt_vertex1 = vertex_shader_program.compile(*context.device);
	auto opt_vertex2 = vertex_shader_program.compile(*context.device);

	auto opt_rendering_fragment = ShaderProgram(
		rendering_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).compile(*context.device);

	auto opt_compositing_fragment = ShaderProgram(
		compositing_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).compile(*context.device);

	if (!opt_vertex1 || !opt_vertex2 || !opt_rendering_fragment || !opt_compositing_fragment)
		throw std::runtime_error("Failed to compile shader programs");

	vk::raii::ShaderModule vertex1 = std::move(*opt_vertex1);
	vk::raii::ShaderModule vertex2 = std::move(*opt_vertex2);
	vk::raii::ShaderModule rendering_fragment = std::move(*opt_rendering_fragment);
	vk::raii::ShaderModule compositing_fragment = std::move(*opt_compositing_fragment);

	// RENDERING RENDER PASS

	// Create the image
	// TODO: resize handler...
	// TODO: create an editor render layer, and embed the objectifier into
	// it...
	rendering.extent = context.extent;
	rendering.image = ImageData(*context.phdev, *context.device,
		format, context.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment
			| vk::ImageUsageFlagBits::eTransferSrc,
		// vk::ImageLayout::eUndefined,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	// Create the staging buffer
	vk::DeviceSize size = 2 * sizeof(uint32_t) * context.extent.width * context.extent.height;
	rendering.staging_buffer = BufferData(*context.phdev, *context.device,
		size,
		vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Create the depth buffer
	// TODO: why do we need this?
	rendering.depth_buffer = DepthBuffer {
		*context.phdev, *context.device,
		vk::Format::eD32Sfloat,
		context.extent
	};

	// Create the render pass
	rendering.render_pass = make_render_pass(*context.device,
		{format},
		{vk::AttachmentLoadOp::eClear},
		rendering.depth_buffer.format,
		vk::AttachmentLoadOp::eClear
	);

	// Create the framebuffer
	vk::ImageView attachments[] = {
		*rendering.image.view,
		*rendering.depth_buffer.view
	};

	vk::FramebufferCreateInfo framebuffer_info {
		{}, *rendering.render_pass,
		2, &attachments[0],
		context.extent.width, context.extent.height, 1
	};

	rendering.framebuffer = vk::raii::Framebuffer(*context.device,
		framebuffer_info
	);

	// Push constants and pipeline layout
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	rendering.ppl = vk::raii::PipelineLayout {
		*context.device,
		{{}, {}, push_constants}
	};

	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo grp_info1 {
		*context.device, rendering.render_pass,
		std::move(vertex1), nullptr,
		std::move(rendering_fragment), nullptr,
		vertex_binding, vertex_attributes,
		rendering.ppl
	};

	grp_info1.blend_attachments = { false };
	grp_info1.cull_mode = vk::CullModeFlagBits::eNone;

	rendering.pipeline = make_graphics_pipeline(grp_info1);

	// COMPOSITING RENDER PASS

	// Render pass
	compositing.render_pass = make_render_pass(
		*context.device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eLoad}, // We don't clear the swapchain
		context.depth_format,
		vk::AttachmentLoadOp::eClear // TODO: shouldnt matter, since
						// it runs last
	);

	// TODO: highlight after all the depth stuff is done (otherwise
	// highlight shows over everything else)

	// Create the pipeline
	vk::PushConstantRange push_constants2 {
		vk::ShaderStageFlagBits::eFragment,
		sizeof(PushConstants), sizeof(CompositingData)
	};

	std::array <vk::PushConstantRange, 2> compositing_push_constants {
		push_constants, push_constants2
	};

	compositing.ppl = vk::raii::PipelineLayout {
		*context.device,
		{{}, {}, compositing_push_constants}
	};

	GraphicsPipelineInfo grp_info2 {
		*context.device, compositing.render_pass,
		std::move(vertex2), nullptr,
		std::move(compositing_fragment), nullptr,
		vertex_binding, vertex_attributes,
		compositing.ppl
	};

	grp_info2.cull_mode = vk::CullModeFlagBits::eNone;

	compositing.pipeline = make_graphics_pipeline(grp_info2);
}

// Render entities and download the image
void Objectifier::render(const vk::raii::CommandBuffer &cmd,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{
	// Default render area (viewport and scissor)
	RenderArea ra {{-1, -1}, {-1, -1}};
	ra.apply(cmd, rendering.image.extent);

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
			*rendering.render_pass,
			*rendering.framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				rendering.image.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Bind the pipeline
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *rendering.pipeline);

	// Setup push constants
	PushConstants pc {
		.view = camera.view_matrix(transform),
		.projection = camera.perspective_matrix()
	};

	// Render all entities (with rasterizers)
	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <Renderable> (i)) {
			pc.model = ecs.get <Transform> (i).matrix();

			// Bind and draw
			const Renderable &rasterizer = ecs.get <Renderable> (i);

			int submeshes = rasterizer.size();
			for (int j = 0; j < submeshes; j++) {
				pc.id = {i, j};
				cmd.pushConstants <PushConstants> (*rendering.ppl,
					vk::ShaderStageFlagBits::eVertex,
					0, pc
				);

				cmd.bindVertexBuffers(0, *rasterizer.get_vertex_buffer(j).buffer, {0});
				cmd.bindIndexBuffer(*rasterizer.get_index_buffer(j).buffer,
					0, vk::IndexType::eUint32
				);

				cmd.drawIndexed(rasterizer.get_index_count(j), 1, 0, 0, 0);
			}
		}
	}

	// End render pass
	cmd.endRenderPass();

	// Download the image
	rendering.image.transition_layout(cmd,
		vk::ImageLayout::eTransferSrcOptimal
	);

	copy_image_to_buffer(cmd,
		rendering.image.image, rendering.staging_buffer.buffer,
		rendering.image.format, rendering.image.extent.width,
		rendering.image.extent.height
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
	rendering.image.transition_layout(cmd,
		vk::ImageLayout::ePresentSrcKHR
	);
}

// Composite a highlighting effect
void Objectifier::composite_highlight(
		const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const vk::Extent2D &extent,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform,
		const std::pair <uint32_t, uint32_t> &id)
{
	// Default render area (viewport and scissor)
	RenderArea ra {{-1, -1}, {-1, -1}};
	ra.apply(cmd, extent);

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
			*compositing.render_pass,
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

	// Bind the pipeline
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *compositing.pipeline);

	// Setup push constants
	PushConstants pc {
		.view = camera.view_matrix(transform),
		.projection = camera.perspective_matrix()
	};

	CompositingData data {
		.expected = {id.first, id.second}
	};

	// Get the entity and submesh
	const Transform &t = ecs.get <Transform> (id.first);
	const Renderable &renderable = ecs.get <Renderable> (id.first);

	pc.model = t.matrix();
	pc.id = {id.first, id.second};

	cmd.pushConstants <PushConstants> (
		*compositing.ppl,
		vk::ShaderStageFlagBits::eVertex,
		0, pc
	);

	cmd.pushConstants <CompositingData> (
		*compositing.ppl,
		vk::ShaderStageFlagBits::eFragment,
		sizeof(PushConstants), data
	);

	cmd.bindVertexBuffers(0, *renderable.get_vertex_buffer(id.second).buffer, {0});
	cmd.bindIndexBuffer(*renderable.get_index_buffer(id.second).buffer,
		0, vk::IndexType::eUint32
	);

	cmd.drawIndexed(renderable.get_index_count(id.second), 1, 0, 0, 0);

	// End render pass
	cmd.endRenderPass();
}

std::pair <uint32_t, uint32_t> Objectifier::query(uint32_t x, uint32_t y)
{
	// Download to host
	rendering.staging_data.resize(rendering.staging_buffer.size/sizeof(uint32_t));
	rendering.staging_buffer.download(rendering.staging_data);

	int index = y * rendering.image.extent.width + x;
	int renderable_id = rendering.staging_data[2 * index];
	int submesh_id = rendering.staging_data[2 * index + 1];
	return {renderable_id, submesh_id};
}

}

}
