#include "../../include/layers/image_renderer.hpp"
#include "../../include/shader_program.hpp"

namespace kobra {

namespace layers {

// Descriptor layout
static const std::vector <DescriptorSetLayoutBinding>
	DESCRIPTOR_SET_LAYOUT_BINDINGS {
	DSLB {
		0,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

// Vertex format
struct _Vertex {
	glm::vec2 position;
	glm::vec2 texcoord;
};

static constexpr vk::VertexInputBindingDescription VERTEX_BINDING {
	0,
	sizeof(_Vertex),
	vk::VertexInputRate::eVertex
};

static const std::vector <vk::VertexInputAttributeDescription>
	VERTEX_ATTRIBUTES {
	vk::VertexInputAttributeDescription {
		0, 0,
		vk::Format::eR32G32Sfloat,
		offsetof(_Vertex, position)
	},
	vk::VertexInputAttributeDescription {
		1, 0,
		vk::Format::eR32G32Sfloat,
		offsetof(_Vertex, texcoord)
	}
};

// Shader source
static constexpr const char *IMAGE_VERTEX_SHADER = R"(
#version 450

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texcoord;
layout (location = 0) out vec2 v_texcoord;

void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
	v_texcoord = texcoord;
}
)";

static constexpr const char *IMAGE_FRAGMENT_SHADER = R"(
#version 450

layout (binding = 0) uniform sampler2D tex;
layout (location = 0) in vec2 v_texcoord;
layout (location = 0) out vec4 frag_color;

void main()
{
	frag_color = texture(tex, vec2(v_texcoord.x, 1.0 - v_texcoord.y));
}
)";

// Constructor
ImageRenderer::ImageRenderer(const Context &context)
		: m_device(context.device),
		m_phdev(context.phdev),
		m_descriptor_pool(context.descriptor_pool),
		m_sync_queue(context.sync_queue)
{
	// Create the render pass
	m_render_pass = make_render_pass(
		*m_device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eLoad},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Create descriptor set layout
	m_dsl = make_descriptor_set_layout(
		*m_device,
		DESCRIPTOR_SET_LAYOUT_BINDINGS
	);

	std::vector <vk::DescriptorSetLayout> dsls {*m_dsl};
	m_dset = std::move(vk::raii::DescriptorSets {
		*m_device,
		{**m_descriptor_pool, dsls}
	}.front());

	// Pipeline layout
	m_ppl  = vk::raii::PipelineLayout(
		*m_device,
		{{}, *m_dsl, {}}
	);

	// Load the shaders
	auto opt_vertex_shader = ShaderProgram(
		IMAGE_VERTEX_SHADER,
		vk::ShaderStageFlagBits::eVertex
	).compile(*m_device);

	auto opt_fragment_shader = ShaderProgram(
		IMAGE_FRAGMENT_SHADER,
		vk::ShaderStageFlagBits::eFragment
	).compile(*m_device);

	if (!opt_vertex_shader || !opt_fragment_shader)
		throw std::runtime_error("Failed to compile shaders");

	// Pipeline creation
	GraphicsPipelineInfo grp_info {
		*m_device,
		m_render_pass,
		std::move(*opt_vertex_shader), nullptr,
		std::move(*opt_fragment_shader), nullptr,
		VERTEX_BINDING, VERTEX_ATTRIBUTES,
		m_ppl
	};

	grp_info.cull_mode = vk::CullModeFlagBits::eNone;
	m_pipeline = make_graphics_pipeline(grp_info);

	// Buffer data for vertices
	// (2 triangles, without index buffer)
	std::vector <_Vertex> vertices {
		{{-1.0f, -1.0f}, {0.0f, 0.0f}},
		{{ 1.0f, -1.0f}, {1.0f, 0.0f}},
		{{-1.0f,  1.0f}, {0.0f, 1.0f}},

		{{1.0f,  1.0f}, {1.0f, 1.0f}},
		{{-1.0f,  1.0f}, {0.0f, 1.0f}},
		{{ 1.0f, -1.0f}, {1.0f, 0.0f}},
	};

	/* m_vertex_buffer = make_buffer(
		*m_device,
		vk::BufferUsageFlagBits::eVertexBuffer,
		vertices
	); */

	m_buffer_data = BufferData(
		*m_phdev, *m_device,
		sizeof(_Vertex) * vertices.size(),
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent
	);

	m_buffer_data.upload(vertices);
}

void ImageRenderer::render(const ImageData &image, const RenderContext &render_context)
{
	// Check if the image is cached
	if (*image.view != m_prev_image_view) {
		m_sampler = make_sampler(*m_device, image);
		m_prev_image_view = *image.view;

		// Add descriptor binding to sync queue
		m_sync_queue->push({
			"ImageRenderer::render binding new image",
			[&]() {
				std::cout << "ImageRenderer::render: Binding new image" << std::endl;
				bind_ds(*m_device, m_dset, m_sampler, image, 0);
			}
		});

		return;
	}

	// TODO: render context method...
	// Apply the rendering area
	render_context.render_area.apply(
		render_context.cmd,
		render_context.extent
	);

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
	render_context.cmd.beginRenderPass(
		vk::RenderPassBeginInfo {
			*m_render_pass,
			*render_context.framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				render_context.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Bind descriptor set and draw
	render_context.cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*m_ppl, 0, *m_dset, {}
	);

	render_context.cmd.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		*m_pipeline
	);

	// Bind vertex buffer
	render_context.cmd.bindVertexBuffers(
		0, *m_buffer_data.buffer, {0}
	);

	// Draw
	render_context.cmd.draw(6, 1, 0, 0);

	render_context.cmd.endRenderPass();
}

}

}
