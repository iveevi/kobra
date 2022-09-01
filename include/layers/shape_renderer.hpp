#ifndef KOBRA_LAYERS_SHAPE_RENDERER_H_
#define KOBRA_LAYERS_SHAPE_RENDERER_H_

// Engine headers
#include "../backend.hpp"
#include "../ui/shapes.hpp"

namespace kobra {

namespace layers {

class ShapeRenderer {
	// Push constants
	struct PushConstants {
		glm::vec2 center;
		float width;
		float height;
		float radius;
		float border_width;
	};

	// Vulkan context
	Context				_ctx;

	const vk::raii::RenderPass	*_render_pass = nullptr;

	// Pipline and descriptor set layout
	vk::raii::Pipeline		_pipeline = nullptr;
	vk::raii::PipelineLayout	_ppl = nullptr;

	// Custom pipelines
	std::list <vk::raii::Pipeline>
					_custom_pipelines;

	// Create new custom pipeline
	vk::raii::Pipeline *_mk_custom_pipeline(ShaderProgram &shader_program) {
		// Load the shaders
		auto vertex_shader = make_shader_module(*_ctx.device, "shaders/bin/ui/basic_vert.spv");
		auto fragment_shader = shader_program.compile(*_ctx.device);
		if (!fragment_shader)
			return nullptr;

		auto pipeline_cache = vk::raii::PipelineCache {*_ctx.device, {}};

		// Vertex binding and attribute descriptions
		auto binding_description = Vertex::vertex_binding();
		auto attribute_descriptions = Vertex::vertex_attributes();

		// Create the graphics pipeline
		GraphicsPipelineInfo grp_info {
			*_ctx.device, *_render_pass,
			std::move(vertex_shader), nullptr,
			std::move(*fragment_shader), nullptr,
			binding_description, attribute_descriptions,
			_ppl
		};

		grp_info.depth_test = false;
		grp_info.depth_write = false;

		_custom_pipelines.push_back(make_graphics_pipeline(grp_info));
		return &_custom_pipelines.back();
	}

	// Vertex type
	struct Vertex {
		glm::vec2 pos;
		glm::vec2 uv;
		glm::vec3 color;

		// Get Vulkan info for vertex
		static vk::VertexInputBindingDescription vertex_binding() {
			return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
		}

		static std::vector <vk::VertexInputAttributeDescription> vertex_attributes() {
			return {
				{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)},
				{1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv)},
				{2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)}
			};
		}
	};

	// Vertices and indices
	BufferData			_b_vertices = nullptr;
	BufferData			_b_indices = nullptr;
public:
	// Default constructor
	ShapeRenderer() = default;

	// Constructor
	ShapeRenderer(const Context &ctx, const vk::raii::RenderPass &render_pass)
			: _ctx(ctx), _render_pass(&render_pass) {
		// Push constants
		auto pcr = vk::PushConstantRange {
			vk::ShaderStageFlagBits::eFragment,
			0, sizeof(PushConstants)
		};

		// Pipline layout
		_ppl = vk::raii::PipelineLayout {
			*_ctx.device,
			{{}, {}, pcr}
		};

		// Pipeline
		auto shaders = make_shader_modules(*_ctx.device, {
			"shaders/bin/ui/basic_vert.spv",
			"shaders/bin/ui/basic_frag.spv"
		});

		auto pipeline_cache = vk::raii::PipelineCache {*_ctx.device, {}};

		// Vertex binding and attribute descriptions
		auto binding_description = Vertex::vertex_binding();
		auto attribute_descriptions = Vertex::vertex_attributes();

		// Create the graphics pipeline
		GraphicsPipelineInfo grp_info {
			*_ctx.device, render_pass,
			std::move(shaders[0]), nullptr,
			std::move(shaders[1]), nullptr,
			binding_description, attribute_descriptions,
			_ppl
		};

		grp_info.depth_test = false;
		grp_info.depth_write = false;

		_pipeline = make_graphics_pipeline(grp_info);

		// Create the buffers
		vk::DeviceSize vsize = sizeof(Vertex) * 1024;
		vk::DeviceSize isize = sizeof(uint32_t) * 1024;

		_b_vertices = BufferData(*_ctx.phdev, *_ctx.device, vsize,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		_b_indices = BufferData(*_ctx.phdev, *_ctx.device, isize,
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);
	}

	// Render
	// TODO: pass extent
	void render(const vk::raii::CommandBuffer &cmd, const std::vector <ui::Rect *> &rects, const RenderArea &ra = {{-1, -1}, {-1, -1}}) {
		// Apply render area
		ra.apply(cmd, _ctx.extent);

		// Gather shapes
		std::vector <Vertex> vertices;
		std::vector <uint32_t> indices;

		// Render to the vectors
		// TODO: helper method for each shape
		// TODO: need some alpha as well

		std::vector <vk::DeviceSize> v_offsets;
		std::vector <vk::DeviceSize> i_offsets;
		std::vector <int> counts;
		std::vector <PushConstants> push_constants;

		for (auto *rect : rects) {
			glm::vec2 min = rect->min;
			glm::vec2 max = rect->max;
			glm::vec2 dim = {_ctx.extent.width, _ctx.extent.height};

			// Conver to NDC
			min = 2.0f * min/dim - 1.0f;
			max = 2.0f * max/dim - 1.0f;

			glm::vec3 color = rect->color;

			std::vector <uint32_t> r_indices {0, 1, 2, 0, 2, 3};
			std::vector <Vertex> r_vertices {
				Vertex {min,				{0, 0}, color},
				Vertex {glm::vec2 {min.x, max.y},	{0, 1}, color},
				Vertex {max,			   	{1, 1}, color},
				Vertex {glm::vec2 {max.x, min.y},	{1, 0}, color}
			};

			// Offsets and centers
			v_offsets.push_back(vertices.size() * sizeof(Vertex));
			i_offsets.push_back(indices.size() * sizeof(uint32_t));
			counts.push_back(r_indices.size());

			PushConstants pc {
				.center = (min + max) / 2.0f,
				.width = (max - min).x,
				.height = (max - min).y,
				.radius = rect->radius,
				.border_width = rect->border_width
			};

			push_constants.push_back(pc);

			// Append to the vectors
			vertices.insert(vertices.end(), r_vertices.begin(), r_vertices.end());
			indices.insert(indices.end(), r_indices.begin(), r_indices.end());

			auto &sp = rect->shader_program;
			if (sp.valid() && !sp.failed() && sp._pipeline == nullptr) {
				KOBRA_LOG_FUNC(Log::WARN) << "Rect has custom shader, creating pipeline for it...\n";
				auto *ppl = _mk_custom_pipeline(sp);
				sp._pipeline = ppl;
			}
		}

		// Upload to the buffers
		_b_vertices.upload(vertices);
		_b_indices.upload(indices);

		// Draw each rectangle
		for (int i = 0; i < rects.size(); i++) {
			// Bind the correct pipeline
			auto &sp = rects[i]->shader_program;
			if (sp.failed())
				continue;

			if (sp.valid()) {
				if (sp._pipeline == nullptr) {
					KOBRA_LOG_FUNC(Log::ERROR) << "Rect has custom shader, but its pipeline is invalid!\n";
					continue;
				}

				// Custom pipeline
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **sp._pipeline);
			} else {
				// Default pipeline
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);
			}

			// Bind the buffers
			cmd.bindVertexBuffers(0, *_b_vertices.buffer, {v_offsets[i]});
			cmd.bindIndexBuffer(*_b_indices.buffer, i_offsets[i], vk::IndexType::eUint32);

			// Bind the push constants
			cmd.pushConstants <PushConstants> (*_ppl,
				vk::ShaderStageFlagBits::eFragment,
				0, push_constants[i]
			);

			// Draw
			cmd.drawIndexed(counts[i], 1, 0, 0, 0);
		}
	}
};

}

}

#endif
