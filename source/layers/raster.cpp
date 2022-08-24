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
		0, sizeof(Rasterizer::PushConstants)
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

	GraphicsPipelineInfo grp_info(*_ctx.device, _render_pass,
		nullptr, nullptr,
		nullptr, nullptr,
		vertex_binding, vertex_attributes,
		_ppl, pipeline_cache
	);

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
		auto box = Mesh::box({0, 0, 0}, {0.5, 0.01, 0.5});

		_area_light = new Rasterizer({_ctx.phdev, _ctx.device}, box, new Material());

		// Setup descriptor set for area light
		_cached_rasterizers.insert(_area_light);

		// Update descriptor set
		_area_light->bind_material(
			{_ctx.phdev, _ctx.device},
			_b_lights,
			[&]() {
				return _make_ds();
			}
		);
	}
}

////////////
// Render //
////////////

void Raster::render(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const ECS &ecs, const RenderArea &ra)
{
	// Apply render area
	ra.apply(cmd, _ctx.extent);

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
	Transform camera_transform;

	bool found_camera = false;

	LightsData lights_data {
		.count = 0,
	};

	std::vector <std::pair <Transform, glm::vec3>> area_light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// Deal with camera component
		if (ecs.exists <Camera> (i)) {
			camera = ecs.get <Camera> (i);
			camera_transform = ecs.get <Transform> (i);
			found_camera = true;
		}

		// Deal with rasterizer component
		if (ecs.exists <Rasterizer> (i)) {
			// Initialize corresponding descriptor
			// set if not done yet
			const Rasterizer *rasterizer = &ecs.get <Rasterizer> (i);

			if (_cached_rasterizers.count(rasterizer) == 0) {
				_cached_rasterizers.insert(rasterizer);
				
				rasterizer->bind_material(
					{_ctx.phdev, _ctx.device},
					_b_lights,
					[&]() {
						return _make_ds();
					}
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
	Rasterizer::PushConstants push_constants {
		.view = camera.view_matrix(camera_transform),
		.perspective = camera.perspective_matrix(),
		.highlight = false,
	};

	// Render all regular meshes
	// TODO: sort by pipeline
	for (int i = 0; i < ecs.size(); i++) {
		if (!ecs.exists <Rasterizer> (i))
			continue;

		// Get transform
		Transform transform = ecs.get <Transform> (i);
		push_constants.model = transform.matrix();

		// Get rasterizer
		const Rasterizer *rasterizer = &ecs.get <Rasterizer> (i);

		// Bind pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*get_pipeline(rasterizer->mode)
		);

		// Draw mesh
		rasterizer->draw(cmd, _ppl, push_constants);
	}

	// Render all area lights
	{
		// Bind plain (albedo) pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*get_pipeline(eAlbedo)
		);

		for (const auto &pr : area_light_transforms) {
			push_constants.model = pr.first.matrix();
			_area_light->materials[0].diffuse = pr.second;
			_area_light->draw(cmd, _ppl, push_constants);
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
