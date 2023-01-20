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

const std::vector <DSLB> Raster::Skybox::dsl_bindings {
	DSLB {
		RASTER_BINDING_SKYBOX,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

//////////////////////
// Static functions //
//////////////////////

// TODO: replace with Mesh::box
static Mesh make_skybox()
{
	static const std::vector <Vertex> vertices {
		Vertex {{-1.0f,  1.0f, -1.0f}},
		Vertex {{-1.0f, -1.0f, -1.0f}},
		Vertex {{1.0f, -1.0f, -1.0f}},
		Vertex {{1.0f, -1.0f, -1.0f}},
		Vertex {{1.0f,  1.0f, -1.0f}},
		Vertex {{-1.0f,  1.0f, -1.0f}},

		Vertex {{-1.0f, -1.0f,  1.0f}},
		Vertex {{-1.0f, -1.0f, -1.0f}},
		Vertex {{-1.0f,  1.0f, -1.0f}},
		Vertex {{-1.0f,  1.0f, -1.0f}},
		Vertex {{-1.0f,  1.0f,  1.0f}},
		Vertex {{-1.0f, -1.0f,  1.0f}},

		Vertex {{1.0f, -1.0f, -1.0f}},
		Vertex {{1.0f, -1.0f,  1.0f}},
		Vertex {{1.0f,  1.0f,  1.0f}},
		Vertex {{1.0f,  1.0f,  1.0f}},
		Vertex {{1.0f,  1.0f, -1.0f}},
		Vertex {{1.0f, -1.0f, -1.0f}},

		Vertex {{-1.0f, -1.0f,  1.0f}},
		Vertex {{-1.0f,  1.0f,  1.0f}},
		Vertex {{1.0f,  1.0f,  1.0f}},
		Vertex {{1.0f,  1.0f,  1.0f}},
		Vertex {{1.0f, -1.0f,  1.0f}},
		Vertex {{-1.0f, -1.0f,  1.0f}},

		Vertex {{-1.0f,  1.0f, -1.0f}},
		Vertex {{1.0f,  1.0f, -1.0f}},
		Vertex {{1.0f,  1.0f,  1.0f}},
		Vertex {{1.0f,  1.0f,  1.0f}},
		Vertex {{-1.0f,  1.0f,  1.0f}},
		Vertex {{-1.0f,  1.0f, -1.0f}},

		Vertex {{-1.0f, -1.0f, -1.0f}},
		Vertex {{-1.0f, -1.0f,  1.0f}},
		Vertex {{1.0f, -1.0f, -1.0f}},
		Vertex {{1.0f, -1.0f, -1.0f}},
		Vertex {{-1.0f, -1.0f,  1.0f}},
		Vertex {{1.0f, -1.0f,  1.0f}}
	};

	static const std::vector <uint32_t> indices {
		0, 1, 2, 2, 3, 0,
		4, 5, 6, 6, 7, 4,
		8, 9, 10, 10, 11, 8,
		12, 13, 14, 14, 15, 12,
		16, 17, 18, 18, 19, 16,
		20, 21, 22, 22, 23, 20
	};

	return std::vector <Submesh> {
		Submesh {vertices, indices, Material {}}
	};
}

////////////////////
// Aux structures //
////////////////////

struct _light {
	alignas(16) glm::vec3 position;
	alignas(16) glm::vec3 intensity;
};

struct _area_light {
	alignas(16) glm::vec3 a;
	alignas(16) glm::vec3 ab;
	alignas(16) glm::vec3 ac;
	alignas(16) glm::vec3 intensity;
};

struct Raster::LightsData {
	int count;
	int n_area_lights;

	// TODO: max lights, not points lights
	struct _light lights[MAX_POINT_LIGHTS];
	struct _area_light area_lights[MAX_POINT_LIGHTS];
};

//////////////////
// Constructors //
//////////////////

Raster::Raster(const Context &ctx, const vk::AttachmentLoadOp &load)
		: _ctx(ctx)
{
	// Start the timer
	_timer.start();

	// Create render pass
	_render_pass = make_render_pass(*ctx.device,
		{ctx.swapchain_format}, {load},
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
		0, sizeof(Renderable::PushConstants)
	};

	// Pipeline layout
	_ppl = vk::raii::PipelineLayout(
		*_ctx.device,
		{{}, *_dsl, push_constants}
	);

	// Pipelines
	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	// TODO: pipeline cache
	GraphicsPipelineInfo grp_info {
		*_ctx.device, _render_pass,
		nullptr, nullptr,
		nullptr, nullptr,
		vertex_binding, vertex_attributes,
		_ppl
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
		auto box = Mesh::box({0, 0, 0}, {0.5, 0.01, 0.5});

		_area_light = new Renderable(_ctx, new Mesh(box));

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

/////////////
// Methods //
/////////////

void Raster::environment_map(const std::string &path)
{
	if (!_skybox.initialized)
		_initialize_skybox();

	_skybox.path = path;
	_skybox.enabled = true;

	// Bind to its descriptor set
	_ctx.texture_loader->bind(_skybox.dset, _skybox.path, RASTER_BINDING_SKYBOX);
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
		.n_area_lights = 0
	};

	// TODO: multiple transform tables
	std::vector <std::pair <Transform, glm::vec3>> area_light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// Deal with camera component
		if (ecs.exists <Camera> (i)) {
			camera = ecs.get <Camera> (i);
			camera_transform = ecs.get <Transform> (i);
			found_camera = true;
		}

		// Deal with rasterizer component
		if (ecs.exists <Renderable> (i)) {
			// Initialize corresponding descriptor
			// set if not done yet
			const Renderable *rasterizer = &ecs.get <Renderable> (i);

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

			/* Update lights data
			_light l {.position = pos, .intensity = light->color * light->power};
			lights_data.lights[lights_data.count++] = l; */

			if (light->type == Light::Type::eArea) {
				glm::vec3 ab = {1, 0, 0};
				glm::vec3 ac = {0, 0, 1};

				ab = transform.apply_vector(ab);
				ac = transform.apply_vector(ac);

				struct _area_light al;
				al.ab = ab/2.0f;
				al.ac = ac/2.0f;
				al.a = transform.position - al.ab - al.ac;
				al.intensity = light->color * light->power;

				lights_data.area_lights[lights_data.n_area_lights++] = al;

				area_light_transforms.push_back({transform, light->color});
			}
		}
	}

	_b_lights.upload(&lights_data, sizeof(lights_data));

	// Render all rasterizer components
	float ms = _timer.elapsed_start();
	Renderable::PushConstants push_constants {
		.time = ms,
		.view = camera.view_matrix(camera_transform),
		.perspective = camera.perspective_matrix(),
		.view_position = camera_transform.position,
		.highlight = false,
	};

	// Render all regular meshes
	// TODO: sort by pipeline
	for (int i = 0; i < ecs.size(); i++) {
		if (!ecs.exists <Renderable> (i))
			continue;

		// Get transform
		Transform transform = ecs.get <Transform> (i);
		push_constants.model = transform.matrix();

		// Get rasterizer
		const Renderable *rasterizer = &ecs.get <Renderable> (i);

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

	// Render skybox
	if (_skybox.enabled) {
		// Bind pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*_skybox.pipeline
		);

		// Bind descriptor set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*_skybox.ppl, 0, *_skybox.dset, {}
		);

		// Push constants
		push_constants.view = glm::mat4(glm::mat3(push_constants.view));
		cmd.pushConstants <Renderable::PushConstants> (
			*_skybox.ppl,
			vk::ShaderStageFlagBits::eVertex,
			0, push_constants
		);

		// Draw
		cmd.bindVertexBuffers(0, *_skybox.vertex_buffer.buffer, {0});
		cmd.draw(36, 1, 0, 0);
	}

	// End
	cmd.endRenderPass();
}

/////////////////////
// Private methods //
/////////////////////

void Raster::_initialize_skybox()
{
	// Create descriptor set layout
	_skybox.dsl = make_descriptor_set_layout(*_ctx.device, Skybox::dsl_bindings);

	// Load shaders
	auto shaders = make_shader_modules(*_ctx.device, {
		"bin/spv/skybox_vert.spv",
		"bin/spv/skybox_frag.spv"
	});

	// Push constants
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(Renderable::PushConstants)
	};

	// Pipeline layout
	_skybox.ppl = vk::raii::PipelineLayout(
		*_ctx.device,
		{{}, *_skybox.dsl, push_constants}
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
		*_ctx.device, _render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		vk::VertexInputBindingDescription {
			0, sizeof(Vertex),
			vk::VertexInputRate::eVertex
		},
		{
			vk::VertexInputAttributeDescription {
				0, 0,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, position)
			},
		},
		_skybox.ppl
	};

/*
		_skybox.ppl, pipeline_cache,
		true, true,
		vk::CullModeFlagBits::eNone
	); */

	grp_info.cull_mode = vk::CullModeFlagBits::eNone;

	// Create pipeline
	_skybox.pipeline = make_graphics_pipeline(grp_info);

	// Create descriptor set
	auto dsets = vk::raii::DescriptorSets {
		*_ctx.device,
		{**_ctx.descriptor_pool, *_skybox.dsl}
	};

	_skybox.dset = std::move(dsets.front());

	// Allocate box buffers
	Submesh box = make_skybox()[0];
	// TODO: use pure float3...

	_skybox.vertex_buffer = BufferData(
		*_ctx.phdev, *_ctx.device,
		box.vertices.size() * sizeof(Vertex),
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	_skybox.index_buffer = BufferData(
		*_ctx.phdev, *_ctx.device,
		box.indices.size() * sizeof(uint32_t),
		vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Upload box data
	_skybox.vertex_buffer.upload(box.vertices);
	_skybox.index_buffer.upload(box.indices);

	// Set initialization status
	_skybox.initialized = true;
}

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

	KOBRA_ASSERT(false, "Renderable: invalid raster mode");
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
