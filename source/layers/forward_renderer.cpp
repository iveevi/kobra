// Engine headers
#include "include/layers/forward_renderer.hpp"
#include "include/renderable.hpp"
#include "include/shader_program.hpp"
#include "source/shaders/bindings.h"

namespace kobra {

namespace layers {

// Push constants
struct PushConstants {
	float		time;

	alignas(16)
	glm::mat4	model;
	glm::mat4	view;
	glm::mat4	perspective;

	alignas(16)
	glm::vec3	view_position;

	// TODO: reorganize this
	float		highlight;
};

//////////////////////
// Static variables //
//////////////////////

static const std::vector <DescriptorSetLayoutBinding>
	BASE_DESCRIPTOR_SET_LAYOUT_BINDINGS {
	DSLB {
		RASTER_BINDING_MATERIAL,
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

	// Create the render pass
	render_pass = make_render_pass(
		*device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	pipeline_packages[BUILTIN_PIPELINE_PACKAGE]
		= *make_pipline_package(
			KOBRA_DIR "/source/shaders/vertex.vert",
			KOBRA_DIR "/source/shaders/color.frag",
			{}, nullptr
		);
}

// Create a new pipeline package
void ForwardRenderer::add_pipeline(
		const std::string &name,
		const std::string &fragment_shader_file,
		const std::vector <DescriptorSetLayoutBinding> &extra_bindings,
		const std::function <void (const vk::raii::DescriptorSet &dset)> &configure_dset)
{
	if (name == BUILTIN_PIPELINE_PACKAGE) {
		KOBRA_LOG_FUNC(Log::WARN)
			<< "Pipeline name \""
			<< BUILTIN_PIPELINE_PACKAGE
			<< "\" is reserved\n";
		return;
	}

	// TODO: at some point allow for custom vertex shaders
	auto ppl_pkg = make_pipline_package(
		KOBRA_DIR "/source/shaders/vertex.vert",
		fragment_shader_file,
		extra_bindings,
		configure_dset
	);

	if (ppl_pkg)
		pipeline_packages[name] = std::move(*ppl_pkg);
}

// Create a descriptor set for the layer
ForwardRenderer::RenderableDset ForwardRenderer::make_renderable_dset
		(PipelinePackage &pipeline_package, uint32_t count)
{
	std::vector <vk::DescriptorSetLayout>
		layouts(count, *pipeline_package.dsl);

	vk::DescriptorSetAllocateInfo alloc_info {
		**descriptor_pool,
		layouts
	};

	auto dsets = vk::raii::DescriptorSets {
		*device,
		alloc_info
	};

	ForwardRenderer::RenderableDset rdset;
	for (auto &d : dsets)
		rdset.emplace_back(std::move(d));

	return rdset;
}

// Configure/update the descriptor set wrt a Renderable component
void ForwardRenderer::configure_renderable_dset
		(const System *system,
                const PipelinePackage &pipeline_package,
		const ForwardRenderer::RenderableDset &dset,
		const Renderable *rasterizer)
{
	assert(dset.size() == rasterizer->material_indices.size());

	auto &materials = rasterizer->material_indices;
	// auto &ubo = rasterizer->ubo;

	for (size_t i = 0; i < dset.size(); ++i) {
		auto &d = dset[i];
		uint32_t material_index = rasterizer->material_indices[i];
		// const Material &m = Material::all[material_index];
                const Material &m = system->get_material(material_index);

		// Bind the textures
		std::string albedo = "blank";
		if (m.has_albedo())
			albedo = m.diffuse_texture;

		std::string normal = "blank";
		if (m.has_normal())
			normal = m.normal_texture;

		loader->bind(d, albedo, RASTER_BINDING_ALBEDO_MAP);
		loader->bind(d, normal, RASTER_BINDING_NORMAL_MAP);

		// Bind material UBO
                KOBRA_LOG_FILE(Log::WARN) << "TODO: bind material UBO\n";
		// bind_ds(*device, d, ubo[i],
		// 	vk::DescriptorType::eUniformBuffer,
		// 	RASTER_BINDING_MATERIAL
		// );

		// If there is a provided configure function, call it
		if (pipeline_package.configure_dset)
			pipeline_package.configure_dset(d);
	}
}

// Render a given scene wrt a given camera
void ForwardRenderer::render
		(const Parameters &parameters,
		const Camera &camera,
		const Transform &camera_transform,
		const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const vk::Extent2D &extent,
		const RenderArea &ra)
{
	// Make sure the pipeline package exists
	if (pipeline_packages.find(parameters.pipeline_package)
			== pipeline_packages.end()) {
		KOBRA_LOG_FUNC(Log::WARN)
			<< "Pipeline package \""
			<< parameters.pipeline_package
			<< "\" does not exist\n";
		return;
	}

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

	// TODO: on dset changes, use sync queue
	PipelinePackage &pipeline_package = pipeline_packages[parameters.pipeline_package];
	for (const auto &[renderable, _] : parameters.renderables) {
		auto &dsets = pipeline_package.dsets;
		if (dsets.find(renderable) == dsets.end()) {
			dsets[renderable] = make_renderable_dset(
				pipeline_package,
				renderable->material_indices.size()
			);

			// Configure the dset
			configure_renderable_dset(parameters.system, pipeline_package, dsets[renderable], renderable);
		}
	}

	// Update the data
	// TODO: update only when needed, and update lights...

	// Bind pipeline
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_package.pipeline);

	// Prepare push constants
	PushConstants pc;

	pc.view = camera.view_matrix(camera_transform);
	pc.perspective = camera.perspective_matrix();
	pc.view_position = camera_transform.position;

	int count = parameters.renderables.size();
	for (int i = 0; i < count; i++) {
		pc.model = std::get <1> (parameters.renderables[i])->matrix();

		const Renderable *renderable = std::get <0> (parameters.renderables[i]);
		ForwardRenderer::RenderableDset &dset = pipeline_package.dsets[renderable];

		int submesh_count = renderable->size();
		for (int j = 0; j < submesh_count; j++) {
			// Push constants
			cmd.pushConstants <PushConstants> (
				*pipeline_package.ppl,
				vk::ShaderStageFlagBits::eVertex,
				0, pc
			);

			// Bind buffers?>
			cmd.bindVertexBuffers(0, *renderable->vertex_buffer[j].buffer, {0});
			cmd.bindIndexBuffer(*renderable->index_buffer[j].buffer,
				0, vk::IndexType::eUint32
			);

			// Bind descriptor set
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				*pipeline_package.ppl, 0, *dset[j], {}
			);

			// Draw
			cmd.drawIndexed(renderable->index_count[j], 1, 0, 0, 0);
		}
	}

	// Environment map, if provided
	if (!parameters.environment_map.empty()) {
		// Load if not loaded
		if (m_skybox.environment_map != parameters.environment_map) {
			configure_environment_map(parameters.environment_map);
			m_skybox.environment_map = parameters.environment_map;
		}

		// Push constants
		cmd.pushConstants <PushConstants> (
			*pipeline_package.ppl,
			vk::ShaderStageFlagBits::eVertex,
			0, pc
		);

		// Bind descriptor set
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
			*m_skybox.ppl, 0, *m_skybox.dset, {}
		);

		// Bind pipeline
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_skybox.pipeline);

		// Load vertex and index buffer and draw
		cmd.bindVertexBuffers(0, *m_skybox.vbo.buffer, {0});
		cmd.bindIndexBuffer(*m_skybox.ibo.buffer, 0, vk::IndexType::eUint32);
		cmd.draw(36, 1, 0, 0);
	}

	// End the render pass
	cmd.endRenderPass();
}

// Private helpers methods
std::optional <ForwardRenderer::PipelinePackage>
ForwardRenderer::make_pipline_package
		(const std::string &vertex_shader_file,
		const std::string &fragment_shader_file,
		const std::vector <DescriptorSetLayoutBinding> &extra_bindings,
		const std::function <void (const vk::raii::DescriptorSet &dset)> &configure_dset)
{
	// Create the built in pipeline package
	PipelinePackage package;

	// Make sure there are no binding conflicts
	for (const auto &binding : extra_bindings) {
		int bind_index = std::get <0> (binding);
		for (const auto &base_binding : BASE_DESCRIPTOR_SET_LAYOUT_BINDINGS) {
			if (std::get <0> (base_binding) == bind_index) {
				// TODO: show index...
				KOBRA_LOG_FUNC(Log::ERROR)
					<< "Binding conflict between base and extra bindings: "
					<< std::get <0> (binding) << std::endl;
				return {};
			}
		}
	}

	// Create descriptor set layout
	std::vector <DescriptorSetLayoutBinding> bindings = BASE_DESCRIPTOR_SET_LAYOUT_BINDINGS;
	bindings.insert(bindings.end(), extra_bindings.begin(), extra_bindings.end());

	package.dsl = make_descriptor_set_layout(*device, bindings);

	// Load the shaders
	const std::string vertex_shader_source = common::read_file(vertex_shader_file);
	const std::string fragment_shader_source = common::read_file(fragment_shader_file);

	ShaderProgram vertex_shader {
		vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	};

	ShaderProgram fragment_shader {
		fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	};

	//
	vk::PushConstantRange push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	// Pipeline layout
	package.ppl  = vk::raii::PipelineLayout(
		*device,
		{{}, *package.dsl, push_constants}
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
		package.ppl
	};

	grp_info.vertex_shader = std::move(*vertex_shader.compile(*device));
	grp_info.fragment_shader = std::move(*fragment_shader.compile(*device));
	// grp_info.polygon_mode = vk::PolygonMode::eFillRectangleNV;
	grp_info.cull_mode = vk::CullModeFlagBits::eNone;

	package.pipeline = make_graphics_pipeline(grp_info);

	// Other properties
	package.configure_dset = configure_dset;

	// Return the package
	return package;
}

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
		Submesh {vertices, indices, 0}
	};
}

// Configure the environment map
void ForwardRenderer::configure_environment_map(const std::string &path)
{
	// Load the texture and sampler
	const ImageData &image = loader->load_texture(path);
	const vk::raii::Sampler &sampler = loader->load_sampler(path);

	// Construct the pipeline if it doesn't exist
	if (!m_skybox.initialized) {
		m_skybox.dsl = make_descriptor_set_layout(
			*device,
			{
				DSLB {
					RASTER_BINDING_SKYBOX,
					// TODO: be more conversative with
					// binginds (e.g. set to 0 in this
					// case...)
					vk::DescriptorType::eCombinedImageSampler,
					1, vk::ShaderStageFlagBits::eFragment
				}
			}
		);

		std::vector <vk::DescriptorSetLayout> layouts(1, *m_skybox.dsl);
		m_skybox.dset = std::move(
			vk::raii::DescriptorSets {
				*device,
				{**descriptor_pool, layouts}
			}.front()
		);

		// Push constants
		vk::PushConstantRange push_constants {
			vk::ShaderStageFlagBits::eVertex,
			0, sizeof(PushConstants)
		};

		// Pipeline layout
		m_skybox.ppl = vk::raii::PipelineLayout(
			*device,
			{{}, *m_skybox.dsl, push_constants}
		);

		// Pipeline cache
		vk::raii::PipelineCache pipeline_cache {
			*device,
			vk::PipelineCacheCreateInfo()
		};

		// Load shader programs
		const std::string vertex_shader_source = common::read_file(KOBRA_DIR "/source/shaders/skybox.vert");
		const std::string fragment_shader_source = common::read_file(KOBRA_DIR "/source/shaders/skybox.frag");

		ShaderProgram vertex_shader {
			vertex_shader_source,
			vk::ShaderStageFlagBits::eVertex
		};

		ShaderProgram fragment_shader {
			fragment_shader_source,
			vk::ShaderStageFlagBits::eFragment
		};

		// Pipelines
		auto vertex_binding = Vertex::vertex_binding();
		auto vertex_attributes = Vertex::vertex_attributes();

		GraphicsPipelineInfo grp_info {
			*device, render_pass,
			nullptr, nullptr,
			nullptr, nullptr,
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
			m_skybox.ppl
		};

		grp_info.cull_mode = vk::CullModeFlagBits::eNone;
		grp_info.vertex_shader = std::move(*vertex_shader.compile(*device));
		grp_info.fragment_shader = std::move(*fragment_shader.compile(*device));

		// Create pipeline
		m_skybox.pipeline = make_graphics_pipeline(grp_info);

		// Create the vertex buffer for the skybox
		// Mesh box = Mesh::box({0, 0, 0}, {1, 1, 1});
		Mesh box = make_skybox();

		m_skybox.vbo = BufferData(
			*phdev, *device,
			box[0].vertices.size() * sizeof(Vertex),
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		m_skybox.ibo = BufferData(
			*phdev, *device,
			box[0].indices.size() * sizeof(uint32_t),
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		m_skybox.vbo.upload(box[0].vertices);
		m_skybox.ibo.upload(box[0].indices);

		// Set the initialized flag
		m_skybox.initialized = true;
	}

	// (Re)bind the texture
	// TODO: sync queue
	bind_ds(*device, m_skybox.dset, sampler, image, RASTER_BINDING_SKYBOX);
}

}

}
