#ifndef KOBRA_LAYERS_HYBRID_TRACER_H_
#define KOBRA_LAYERS_HYBRID_TRACER_H_

// Engine headers
#include "../backend.hpp"
#include "../vertex.hpp"

// TODO: remove
#include "../../shaders/raster/bindings.h"
#include "../camera.hpp"
#include "../ecs.hpp"
#include "../texture_manager.hpp"
#include "../transform.hpp"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;

namespace layers {

// Hybrid ray/path tracer:
//	Rasterizes the scene to get the G-buffer, which is then used for ray/path
//	tracing and producing effects like GI and reflections.
struct HybridTracer {
	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;

	// Geometry buffers
	ImageData positions = nullptr;
	ImageData normals = nullptr;

	// Material buffers
	ImageData albedo = nullptr;
	ImageData specular = nullptr;
	ImageData extra = nullptr;

	// Depth buffer
	DepthBuffer depth = nullptr;

	// Vulkan structures
	RenderPass render_pass = nullptr;
	Framebuffer framebuffer = nullptr;

	Pipeline pipeline = nullptr;
	PipelineLayout ppl = nullptr;

	vk::Extent2D extent = { 0, 0 };

	// Descriptor set bindings
	static const std::vector <DSLB> dsl_bindings;

	// Descriptor set layout
	vk::raii::DescriptorSetLayout dsl = nullptr;

	// Descriptor sets
	using RasterizerDset = std::vector <vk::raii::DescriptorSet>;

	std::map <const Rasterizer *, RasterizerDset> dsets;

	// Functions
	static HybridTracer make(const Context &);
};

// Static member variables
const std::vector <DSLB> HybridTracer::dsl_bindings {
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
	}
};

// Allocate the framebuffer images
void allocate_framebuffer_images(HybridTracer &layer, const Context &context, const vk::Extent2D &extent)
{
	// Formats for each framebuffer image
	static vk::Format fmt_positions = vk::Format::eR32G32B32A32Sfloat;
	static vk::Format fmt_normals = vk::Format::eR32G32B32A32Sfloat;
	static vk::Format fmt_albedo = vk::Format::eR32G32B32A32Sfloat;
	static vk::Format fmt_specular = vk::Format::eR32G32B32A32Sfloat;
	static vk::Format fmt_extra = vk::Format::eR32G32B32A32Sfloat;

	// Other image propreties
	static vk::MemoryPropertyFlags mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
	static vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
	static vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	static vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
	static vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment
		| vk::ImageUsageFlagBits::eTransferSrc;

	// Create the images
	layer.positions = ImageData {
		*context.phdev, *context.device,
		fmt_positions, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.normals = ImageData {
		*context.phdev, *context.device,
		fmt_normals, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.albedo = ImageData {
		*context.phdev, *context.device,
		fmt_albedo, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.specular = ImageData {
		*context.phdev, *context.device,
		fmt_specular, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.extra = ImageData {
		*context.phdev, *context.device,
		fmt_extra, extent, tiling,
		usage, layout, mem_flags, aspect
	};
}

// Push constants
struct PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
};

// Create the layer
HybridTracer HybridTracer::make(const Context &context)
{
	// To return
	HybridTracer layer;

	// Extract critical Vulkan structures
	layer.device = context.device;
	layer.phdev = context.phdev;
	layer.descriptor_pool = context.descriptor_pool;

	// Create the framebuffers
	layer.extent = context.extent;

	allocate_framebuffer_images(layer, context, layer.extent);

	layer.depth = DepthBuffer {
		*context.phdev, *context.device,
		vk::Format::eD32Sfloat, context.extent
	};

	// Create the render pass
	auto eClear = vk::AttachmentLoadOp::eClear;

	layer.render_pass = make_render_pass(*context.device,
		{
			layer.positions.format,
			layer.normals.format,
			layer.albedo.format,
			layer.specular.format,
			layer.extra.format
		},
		{eClear, eClear, eClear, eClear, eClear},
		layer.depth.format,
		eClear
	);

	// Create the framebuffer
	std::vector <vk::ImageView> attachments {
		*layer.positions.view,
		*layer.normals.view,
		*layer.albedo.view,
		*layer.specular.view,
		*layer.extra.view,
		*layer.depth.view
	};

	vk::FramebufferCreateInfo fb_info {
		{}, *layer.render_pass,
		(uint32_t) attachments.size(),
		attachments.data(),
		context.extent.width,
		context.extent.height,
		1
	};

	layer.framebuffer = Framebuffer {*context.device, fb_info};

	// Descriptor set layout
	layer.dsl = make_descriptor_set_layout(*context.device, dsl_bindings);

	// Push constants and pipeline layout
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	layer.ppl = PipelineLayout {
		*context.device,
		{{}, *layer.dsl, push_constants}
	};

	// Create the pipeline
	auto shaders = make_shader_modules(*context.device, {
		"bin/spv/hybrid_deferred_vert.spv",
		"bin/spv/hybrid_deferred_frag.spv"
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

	grp_info.color_blend_attachments = 5;

	layer.pipeline = make_graphics_pipeline(grp_info);

	// Return
	return layer;
}

// Create a descriptor set for the layer
HybridTracer::RasterizerDset serve_dset(HybridTracer &layer, uint32_t count)
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

	HybridTracer::RasterizerDset rdset;
	for (auto &d : dsets)
		rdset.emplace_back(std::move(d));

	return rdset;
}

// Configure/update the descriptor set wrt a Rasterizer component
void configure_dset(HybridTracer &layer,
		const HybridTracer::RasterizerDset &dset,
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

// Render the deferred stage (generate the G-buffer)

// TODO: perform this in a separate command buffer than the main one used to
// present, etc (and separate queue)
void render(HybridTracer &layer,
		const CommandBuffer &cmd,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{

	// Preprocess the entities
	std::vector <const Rasterizer *> rasterizers;

	for (int i = 0; i < ecs.size(); i++) {
		// TODO: one unifying renderer component, with options for
		// raytracing, etc
		if (ecs.exists <Rasterizer> (i)) {
			const auto *rasterizer = &ecs.get <Rasterizer> (i);
			rasterizers.push_back(rasterizer);

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
	}

	// Default render area (viewport and scissor)
	RenderArea ra {{-1, -1}, {-1, -1}};
	ra.apply(cmd, layer.extent);

	// Clear colors
	// TODO: easier function to use
	vk::ClearValue color_clear {
		std::array <float, 4> {0.0f, 0.0f, 0.0f, 0.0f}
	};

	std::array <vk::ClearValue, 6> clear_values {
		color_clear, color_clear, color_clear,
		color_clear, color_clear,
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
				layer.extent
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

	// Render all entities with a rasterizer component
	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <Rasterizer> (i)) {
			pc.model = ecs.get <Transform> (i).matrix();

			cmd.pushConstants <PushConstants> (*layer.ppl,
				vk::ShaderStageFlagBits::eVertex,
				0, pc
			);

			// Bind and draw
			const Rasterizer &rasterizer = ecs.get <Rasterizer> (i);
			const HybridTracer::RasterizerDset &dset = layer.dsets[&rasterizer];

			int submeshes = rasterizer.size();
			for (int i = 0; i < submeshes; i++) {
				// Bind buffers
				cmd.bindVertexBuffers(0, *rasterizer.get_vertex_buffer(i).buffer, {0});
				cmd.bindIndexBuffer(*rasterizer.get_index_buffer(i).buffer,
					0, vk::IndexType::eUint32
				);

				// Bind descriptor set
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
					*layer.ppl, 0, *dset[i], {}
				);

				// Draw
				cmd.drawIndexed(rasterizer.get_index_count(i), 1, 0, 0, 0);
			}
		}
	}

	// End render pass
	cmd.endRenderPass();
}

}

}

#endif
