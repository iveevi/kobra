// Engine headers
#include "../../include/layers/raster.hpp"
#include "../../shaders/raster/bindings.h"

namespace kobra {

namespace layers {

//////////////////////
// Static variables //
//////////////////////

const std::vector <DSLB> Raster::_dsl_bindings {
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

////////////////////
// Aux structures //
////////////////////

struct Raster::PushConstants {
	glm::mat4	model;
	glm::mat4	view;
	glm::mat4	perspective;

	glm::vec3	albedo;
	int		type;
	float		highlight;
	float		has_albedo;
	float		has_normal;
};


struct _light {
	alignas(16)
	glm::vec3	position;

	alignas(16)
	glm::vec3	intensity;
};

struct Raster::LightsData {
	int count;

	// TODO: max lights, not points lights
	_light lights[MAX_POINT_LIGHTS];
	// aligned_vec4 positions[MAX_POINT_LIGHTS];
};

//////////////////
// Constructors //
//////////////////

Raster::Raster(const Context &ctx, const vk::AttachmentLoadOp &load)
		: _ctx(ctx)
{
	// Create render pass
	_render_pass = make_render_pass(*ctx.device,
		ctx.swapchain_format,
		ctx.depth_format, load
	);

	// Create descriptor set layout
	_dsl = make_descriptor_set_layout(*_ctx.device, _dsl_bindings);

	// Load all shaders
	auto shaders = make_shader_modules(*_ctx.device, {
		"shaders/bin/raster/vertex.spv",
		"shaders/bin/raster/color_frag.spv",
		"shaders/bin/raster/normal_frag.spv",
		"shaders/bin/raster/blinn_phong_frag.spv"
	});

	// Push constants
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	// Pipeline layout
	_ppl = vk::raii::PipelineLayout(
		*_ctx.device,
		{{}, *_dsl, push_constants}
	);

	// Pipeline cache
	vk::raii::PipelineCache pipeline_cache {
		*_ctx.device,
		vk::PipelineCacheCreateInfo()
	};

	// Pipelines
	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo grp_info {
		.device = *_ctx.device,
		.render_pass = _render_pass,

		.vertex_shader = nullptr,
		.fragment_shader = nullptr,

		.vertex_binding = vertex_binding,
		.vertex_attributes = vertex_attributes,

		.pipeline_layout = _ppl,
		.pipeline_cache = pipeline_cache,

		.depth_test = true,
		.depth_write = true
	};

	// Common vertex shader
	grp_info.vertex_shader = std::move(shaders[0]);

	// Albedo
	grp_info.fragment_shader = std::move(shaders[1]);
	_p_albedo = make_graphics_pipeline(grp_info);

	// Normals
	grp_info.fragment_shader = std::move(shaders[2]);
	_p_normal = make_graphics_pipeline(grp_info);

	// Blinn-Phong
	grp_info.fragment_shader = std::move(shaders[3]);
	_p_phong = make_graphics_pipeline(grp_info);

	// Create buffer for lights
	_b_lights = BufferData(*_ctx.phdev, *_ctx.device, sizeof(LightsData),
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Create box for rendering area lights
	{
		auto box = Mesh::box({0, 0, 0}, {0.5, 0.5, 0.5});

		_area_light = new Rasterizer({_ctx.phdev, _ctx.device}, box, new Material());

		// Setup descriptor set for area light
		_ds_components.insert({_area_light, _make_ds()});
		const auto &ds = _ds_components.at(_area_light);

		Device dev {
			_ctx.phdev,
				_ctx.device
		};

		// Update descriptor set
		_area_light->bind_material(dev, ds);

		// Bind lights buffer
		bind_ds(*_ctx.device, ds, _b_lights,
			vk::DescriptorType::eUniformBuffer,
			RASTER_BINDING_POINT_LIGHTS
		);
	}
}

////////////
// Render //
////////////

void Raster::render(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const ECS &ecs)
{
	// Set viewport
	cmd.setViewport(0,
		vk::Viewport {
			0.0f, 0.0f,
			static_cast <float> (_ctx.extent.width),
			static_cast <float> (_ctx.extent.height),
			0.0f, 1.0f
		}
	);

	// Set scissor
	cmd.setScissor(0,
		vk::Rect2D {
			vk::Offset2D {0, 0},
			_ctx.extent
		}
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
	cmd.beginRenderPass(
		vk::RenderPassBeginInfo {
			*_render_pass,
			*framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				_ctx.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Initialization phase
	Camera camera;
	bool found_camera = false;

	LightsData lights_data {
		.count = 0,
	};

	std::vector <std::pair <Transform, glm::vec3>> area_light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// Deal with camera component
		if (ecs.exists <Camera> (i)) {
			camera = ecs.get <Camera> (i);
			found_camera = true;
		}

		// Deal with rasterizer component
		if (ecs.exists <Rasterizer> (i)) {
			// Initialize corresponding descriptor
			// set if not done yet
			const Rasterizer *rasterizer = &ecs.get <Rasterizer> (i);
			if (_ds_components.count(rasterizer) == 0) {
				_ds_components.insert({rasterizer, _make_ds()});
				const auto &ds = _ds_components.at(rasterizer);

				Device dev {
					_ctx.phdev,
					_ctx.device
				};

				// Update descriptor set
				rasterizer->bind_material(dev, ds);

				// Bind lights buffer
				bind_ds(*_ctx.device, ds, _b_lights,
					vk::DescriptorType::eUniformBuffer,
					RASTER_BINDING_POINT_LIGHTS
				);
			}
		}

		// Deal with light component
		if (ecs.exists <Light> (i)) {
			// Initialize corresponding descriptor
			// set if not done yet
			const Light *light = &ecs.get <Light> (i);

			// Transform
			const auto &transform = ecs.get <Transform> (i);
			auto pos = transform.position;

			// Update lights data
			_light l {.position = pos, .intensity = light->color * light->power};
			lights_data.lights[lights_data.count++] = l;

			if (light->type == Light::Type::eArea)
				area_light_transforms.push_back({transform, light->color});
		}
	}

	_b_lights.upload(&lights_data, sizeof(lights_data));

	// Render all rasterizer components
	PushConstants push_constants {
		.view = camera.view(),
		.perspective = camera.perspective(),
		.type = Shading::eDiffuse,
		.highlight = false,
		.has_albedo = false,
		.has_normal = false
	};

	// Render all regular meshes
	for (int i = 0; i < ecs.size(); i++) {
		if (!ecs.exists <Rasterizer> (i))
			continue;

		// Get transform
		Transform transform = ecs.get <Transform> (i);
		push_constants.model = transform.matrix();

		// Get rasterizer
		const Rasterizer *rasterizer = &ecs.get <Rasterizer> (i);
		const auto &ds = _ds_components.at(rasterizer);

		// Bind pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*get_pipeline(rasterizer->mode)
		);

		// Bind descriptor set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*_ppl, 0, *ds, {}
		);

		// Bind vertex and index buffers
		rasterizer->bind_buffers(cmd);

		// Update push constants
		push_constants.albedo = rasterizer->material->diffuse;
		push_constants.has_albedo = rasterizer->material->has_albedo();
		push_constants.has_normal = rasterizer->material->has_normal();

		// Push constant
		cmd.pushConstants <PushConstants> (
			*_ppl, vk::ShaderStageFlagBits::eVertex,
			0, push_constants
		);

		// Draw
		cmd.drawIndexed(rasterizer->indices, 1, 0, 0, 0);
	}

	// Render all area lights
	{
		// Bind plain (albedo) pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*get_pipeline(eAlbedo)
		);

		// Update push constants
		push_constants.has_albedo = false;
		push_constants.has_normal = false;

		// Bind descriptor set
		const auto &ds = _ds_components.at(_area_light);

		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*_ppl, 0, *ds, {}
		);

		// Bind vertex and index buffers
		_area_light->bind_buffers(cmd);

		for (const auto &pr: area_light_transforms) {
			push_constants.model = pr.first.matrix();
			push_constants.albedo = pr.second;

			// Push constant
			cmd.pushConstants <PushConstants> (
				*_ppl, vk::ShaderStageFlagBits::eVertex,
				0, push_constants
			);

			// Draw
			cmd.drawIndexed(_area_light->indices, 1, 0, 0, 0);
		}
	}

	// End
	cmd.endRenderPass();
}

/////////////////////
// Private methods //
/////////////////////

const vk::raii::Pipeline &Raster::get_pipeline(RasterMode mode)
{
	switch (mode) {
	case RasterMode::eAlbedo:
		return _p_albedo;
	case RasterMode::eNormal:
		return _p_normal;
	case RasterMode::ePhong:
		return _p_phong;
	default:
		break;
	}

	KOBRA_ASSERT(false, "Rasterizer: invalid raster mode");
}

vk::raii::DescriptorSet	Raster::_make_ds() const
{
	auto dsets = vk::raii::DescriptorSets {
		*_ctx.device,
		{**_ctx.descriptor_pool, *_dsl}
	};

	return std::move(dsets.front());
}

}

}
