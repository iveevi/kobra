// Engine headers
#include "../../include/layers/forward_renderer.hpp"
#include "../../include/renderable.hpp"
#include "../../shaders/raster/bindings.h"
#include "../../include/ecs.hpp"

namespace kobra {

namespace layers {

//////////////////////
// Static variables //
//////////////////////

const std::vector <DSLB> ForwardRenderer::dsl_bindings {
	DSLB {
		RASTER_BINDING_UBO,
		vk::DescriptorType::eUniformBuffer,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_ALBEDO_MAP,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_NORMAL_MAP,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_POINT_LIGHTS,
		vk::DescriptorType::eUniformBuffer,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

// Create a forward renderer layer
ForwardRenderer::ForwardRenderer(const Context &context)
{
	// Extract critical Vulkan structures
	device = context.device;
	phdev = context.phdev;
	descriptor_pool = context.descriptor_pool;

	loader = context.texture_loader;

	extent = context.extent;

	// Create the render pass
	render_pass = make_render_pass(
		*device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Create descriptor set layout
	dsl = make_descriptor_set_layout(
		*device,
		dsl_bindings
	);

	// Load the default available shaders
	// TODO: store as shader program structs...
	auto shaders = make_shader_modules(
		*device,
		{
			KOBRA_SHADERS_DIR "/vertex_vert.spv",
			KOBRA_SHADERS_DIR "/color_frag.spv"
		}
	);

	//
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(Renderable::PushConstants)
	};

	// Pipeline layout
	ppl  = vk::raii::PipelineLayout(
		*device,
		{{}, *dsl, push_constants}
	);

	// Pipelines
	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo grp_info {
		*device,
		render_pass,
		nullptr, nullptr,
		nullptr, nullptr,
		vertex_binding, vertex_attributes,
		ppl
	};

	grp_info.vertex_shader = std::move(shaders[0]);
	grp_info.fragment_shader = std::move(shaders[1]);

	pipeline = make_graphics_pipeline(grp_info);
}

// Create a descriptor set for the layer
static ForwardRenderer::RenderableDset serve_dset(ForwardRenderer &layer, uint32_t count)
{
	std::vector <vk::DescriptorSetLayout> layouts(count, *layer.dsl);

	vk::DescriptorSetAllocateInfo alloc_info {
		**layer.descriptor_pool,
		layouts
	};

	auto dsets = vk::raii::DescriptorSets {
		*layer.device,
		alloc_info
	};

	ForwardRenderer::RenderableDset rdset;
	for (auto &d : dsets)
		rdset.emplace_back(std::move(d));

	return rdset;
}

// Configure/update the descriptor set wrt a Renderable component
static void configure_dset(ForwardRenderer &layer,
		const ForwardRenderer::RenderableDset &dset,
		const Renderable *rasterizer)
{
	assert(dset.size() == rasterizer->materials.size());

	auto &materials = rasterizer->materials;
	auto &ubo = rasterizer->ubo;

	for (size_t i = 0; i < dset.size(); ++i) {
		auto &d = dset[i];
		auto &m = rasterizer->materials[i];

		// Bind the textures
		std::string albedo = "blank";
		if (materials[i].has_albedo())
			albedo = materials[i].albedo_texture;

		std::string normal = "blank";
		if (materials[i].has_normal())
			normal = materials[i].normal_texture;

		layer.loader->bind(d, albedo, RASTER_BINDING_ALBEDO_MAP);
		layer.loader->bind(d, normal, RASTER_BINDING_NORMAL_MAP);

		// Bind material UBO
		bind_ds(*layer.device, d, ubo[i],
			vk::DescriptorType::eUniformBuffer,
			RASTER_BINDING_UBO
		);
	}
}

// Render a given scene wrt a given camera
void ForwardRenderer::render(const ECS &ecs,
		const Camera &camera,
		const Transform &camera_transform,
		const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const RenderArea &ra)
{
	// Apply the rendering area
	ra.apply(cmd, extent);

	// Clear colors
	// FIXME: seriously make a method from clering values...
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
			*render_pass,
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

	// Preprocess the entities
	std::vector <const Renderable *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// TODO: one unifying renderer component, with options for
		// raytracing, etc
		if (ecs.exists <Renderable> (i)) {
			const auto *rasterizer = &ecs.get <Renderable> (i);
			const auto *transform = &ecs.get <Transform> (i);

			rasterizers.push_back(rasterizer);
			rasterizer_transforms.push_back(transform);

			// If not it the dsets dictionary, create it
			if (dsets.find(rasterizer) == dsets.end()) {
				dsets[rasterizer] = serve_dset(
					*this, // TODO: change signature
					rasterizer->materials.size()
				);

				// Configure the dset
				configure_dset(*this, dsets[rasterizer], rasterizer);
			}
		}

		if (ecs.exists <Light> (i)) {
			const auto *light = &ecs.get <Light> (i);
			const auto *transform = &ecs.get <Transform> (i);

			lights.push_back(light);
			light_transforms.push_back(transform);
		}
	}

	// Update the data
	// TODO: update only when needed, and update lights...

	// Bind pipeline
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

	// Prepare push constants
	Renderable::PushConstants pc;

	pc.view = camera.view_matrix(camera_transform);
	pc.perspective = camera.perspective_matrix();
	pc.view_position = camera_transform.position;

	int count = rasterizers.size();
	for (int i = 0; i < count; i++) {
		pc.model = rasterizer_transforms[i]->matrix();

		const Renderable &rasterizer = *rasterizers[i];
		ForwardRenderer::RenderableDset &dset = dsets[rasterizers[i]];

		int submesh_count = rasterizers[i]->size();
		for (int j = 0; j < submesh_count; j++) {
			// Push constants
			cmd.pushConstants <Renderable::PushConstants> (*ppl,
				vk::ShaderStageFlagBits::eVertex,
				0, pc
			);

			// Bind buffers
			cmd.bindVertexBuffers(0, *rasterizer.get_vertex_buffer(j).buffer, {0});
			cmd.bindIndexBuffer(*rasterizer.get_index_buffer(j).buffer,
				0, vk::IndexType::eUint32
			);

			// Bind descriptor set
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				*ppl, 0, *dset[j], {}
			);

			// Draw
			cmd.drawIndexed(rasterizer.get_index_count(j), 1, 0, 0, 0);
		}
	}

	// End the render pass
	cmd.endRenderPass();
}

}

}
