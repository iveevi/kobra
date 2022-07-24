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

	// Pipline and descriptor set layout
	vk::raii::Pipeline		_pipeline = nullptr;
	vk::raii::PipelineLayout	_ppl = nullptr;

	// Vertex type
	struct Vertex {
		glm::vec2 pos;
		glm::vec3 color;

		// Get Vulkan info for vertex
		static vk::VertexInputBindingDescription
				vertex_binding() {
			return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
		}

		static std::vector <vk::VertexInputAttributeDescription>
				vertex_attributes() {
			return {
				{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)},
				{1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)},
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
			: _ctx(ctx) {
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
			"shaders/bin/gui/basic_vert.spv",
			"shaders/bin/gui/basic_frag.spv"
		});

		auto pipeline_cache = vk::raii::PipelineCache {*_ctx.device, {}};

		// Vertex binding and attribute descriptions
		auto binding_description = Vertex::vertex_binding();
		auto attribute_descriptions = Vertex::vertex_attributes();

		// Create the graphics pipeline
		GraphicsPipelineInfo grp_info {
			.device = *_ctx.device,
			.render_pass = render_pass,

			.vertex_shader = std::move(shaders[0]),
			.fragment_shader = std::move(shaders[1]),

			.vertex_binding = binding_description,
			.vertex_attributes = attribute_descriptions,

			.pipeline_layout = _ppl,
			.pipeline_cache = pipeline_cache,

			.depth_test = false,
			.depth_write = false,
		};

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
	void render(const vk::raii::CommandBuffer &cmd, const std::vector <ui::Rect> &rects) {
		std::vector <Vertex> vertices;
		std::vector <uint32_t> indices;

		// Render to the vectors
		// TODO: helper method for each shape
		// TODO: need some alpha as well

		std::vector <vk::DeviceSize> v_offsets;
		std::vector <vk::DeviceSize> i_offsets;
		std::vector <int> counts;
		std::vector <PushConstants> push_constants;

		for (const auto &rect : rects) {
			glm::vec2 min = rect.min;
			glm::vec2 max = rect.max;
			glm::vec2 dim = {_ctx.extent.width, _ctx.extent.height};

			// Conver to NDC
			min = 2.0f * min/dim - 1.0f;
			max = 2.0f * max/dim - 1.0f;

			glm::vec3 color = rect.color;

			uint32_t i = vertices.size();
			std::vector <uint32_t> r_indices {
				i + 0, i + 1, i + 2,
				i + 2, i + 3, i + 0
			};

			std::vector <Vertex> r_vertices {
				Vertex {min,				color},
				Vertex {glm::vec2 {min.x, max.y},	color},
				Vertex {max,				color},
				Vertex {glm::vec2 {max.x, min.y},	color}
			};

			// Offsets and centers
			v_offsets.push_back(vertices.size() * sizeof(Vertex));
			i_offsets.push_back(indices.size() * sizeof(uint32_t));
			counts.push_back(r_indices.size());

			PushConstants pc {
				.center = (min + max) / 2.0f,
				.width = (max - min).x,
				.height = (max - min).y,
				.radius = rect.radius,
				.border_width = rect.border_width
			};

			push_constants.push_back(pc);

			// Append to the vectors
			vertices.insert(vertices.end(), r_vertices.begin(), r_vertices.end());
			indices.insert(indices.end(), r_indices.begin(), r_indices.end());
		}

		// Upload to the buffers
		_b_vertices.upload(vertices);
		_b_indices.upload(indices);

		// Bind the pipeline
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);

		// Draw each rectangle
		for (int i = 0; i < push_constants.size(); i++) {
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
