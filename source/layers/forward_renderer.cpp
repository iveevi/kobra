// Engine headers
#include "../../include/layers/forward_renderer.hpp"
#include "../../include/renderer.hpp"
#include "../../include/texture_manager.hpp"
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
ForwardRenderer ForwardRenderer::make(const Context &context)
{
	// Layer to return
	ForwardRenderer layer;
	
	// Extract critical Vulkan structures
	layer.device = context.device;
	layer.phdev = context.phdev;
	layer.descriptor_pool = context.descriptor_pool;
	
	layer.extent = context.extent;

	// Create the render pass
	layer.render_pass = make_render_pass(
		*layer.device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Create descriptor set layout
	layer.dsl = make_descriptor_set_layout(
		*layer.device,
		layer.dsl_bindings
	);

	// Load the default available shaders
	// TODO: store as shader program structs...
	auto shaders = make_shader_modules(
		*layer.device,
		{
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/color_frag.spv"
		}
	);

	//
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(Rasterizer::PushConstants)
	};

	// Pipeline layout
	layer.ppl  = vk::raii::PipelineLayout(
		*layer.device,
		{{}, *layer.dsl, push_constants}
	);

	// Pipelines
	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo grp_info {
		*layer.device,
		layer.render_pass,
		nullptr, nullptr,
		nullptr, nullptr,
		vertex_binding, vertex_attributes,
		layer.ppl
	};

	grp_info.vertex_shader = std::move(shaders[0]);
	grp_info.fragment_shader = std::move(shaders[1]);

	layer.pipeline = make_graphics_pipeline(grp_info);

	// Return the layer
	return layer;
}

// Create a descriptor set for the layer
static ForwardRenderer::RasterizerDset serve_dset(ForwardRenderer &layer, uint32_t count)
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

	ForwardRenderer::RasterizerDset rdset;
	for (auto &d : dsets)
		rdset.emplace_back(std::move(d));

	return rdset;
}

// Configure/update the descriptor set wrt a Rasterizer component
static void configure_dset(ForwardRenderer &layer,
		const ForwardRenderer::RasterizerDset &dset,
		const Rasterizer *rasterizer)
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

		TextureManager::bind(
			*layer.phdev, *layer.device,
			d, albedo,
			// TODO: enum like RasterBindings::eAlbedo
			RASTER_BINDING_ALBEDO_MAP
		);

		TextureManager::bind(
			*layer.phdev, *layer.device,
			d, normal,
			RASTER_BINDING_NORMAL_MAP
		);

		// Bind material UBO
		bind_ds(*layer.device, d, ubo[i],
			vk::DescriptorType::eUniformBuffer,
			RASTER_BINDING_UBO
		);
	}
}

// Render a given scene wrt a given camera
void render(ForwardRenderer &layer,
		const ECS &ecs,
		const Camera &camera,
		const Transform &camera_transform,
		const CommandBuffer &cmd,
		const Framebuffer &framebuffer,
		const RenderArea &ra)
{
	// Apply the rendering area
	ra.apply(cmd, layer.extent);
	
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
	
	// Preprocess the entities
	std::vector <const Rasterizer *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// TODO: one unifying renderer component, with options for
		// raytracing, etc
		if (ecs.exists <Rasterizer> (i)) {
			const auto *rasterizer = &ecs.get <Rasterizer> (i);
			const auto *transform = &ecs.get <Transform> (i);

			rasterizers.push_back(rasterizer);
			rasterizer_transforms.push_back(transform);

			// If not it the dsets dictionary, create it
			if (layer.dsets.find(rasterizer) == layer.dsets.end()) {
				layer.dsets[rasterizer] = serve_dset(
					layer,
					rasterizer->materials.size()
				);

				// Configure the dset
				configure_dset(layer, layer.dsets[rasterizer], rasterizer);
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
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *layer.pipeline);

	// Prepare push constants
	Rasterizer::PushConstants pc;

	pc.view = camera.view_matrix(camera_transform);
	pc.perspective = camera.perspective_matrix();
	pc.view_position = camera_transform.position;

	int count = rasterizers.size();
	for (int i = 0; i < count; i++) {
		pc.model = rasterizer_transforms[i]->matrix();

		const Rasterizer &rasterizer = *rasterizers[i];
		ForwardRenderer::RasterizerDset &dset = layer.dsets[rasterizers[i]];

		int submesh_count = rasterizers[i]->size();
		for (int j = 0; j < submesh_count; j++) {
			// Push constants
			cmd.pushConstants <Rasterizer::PushConstants> (*layer.ppl,
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
				*layer.ppl, 0, *dset[j], {}
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