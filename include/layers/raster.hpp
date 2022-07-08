#ifndef KOBRA_LAYERS_RASTERIZER_H_
#define KOBRA_LAYERS_RASTERIZER_H_

// Standard headers
#include <map>

// Engine headers
// TODO: move layer.hpp to this directory
// #include "../layer.hpp"

#include "../backend.hpp"
#include "../vertex.hpp"
#include "../ecs.hpp"

namespace kobra {

namespace layers {

class Raster {
	// Push constant
	struct PushConstant {
		glm::mat4	model;
		glm::mat4	view;
		glm::mat4	projection;

		glm::vec3	albedo;
		int		type;
		float		highlight;
		float		has_albedo;
		float		has_normal;
	};

	// Vulkan context
	Context				_ctx;

	// Other vulkan structures
	vk::raii::RenderPass		_render_pass = nullptr;

	vk::raii::PipelineLayout	_ppl = nullptr;
	vk::raii::Pipeline		_p_albedo = nullptr;
	vk::raii::Pipeline		_p_normal = nullptr;
	vk::raii::Pipeline		_p_phong = nullptr;

	// Descriptor set layout and bindings
	vk::raii::DescriptorSetLayout	_dsl = nullptr;

	static const std::vector <DSLB>	_dsl_bindings;

	// Create a descriptor set
	vk::raii::DescriptorSet	_make_ds() const {
		auto dsets = vk::raii::DescriptorSets {
			_ctx.device,
			{*_ctx.descriptor_pool, *_dsl}
		};

		return std::move(dsets.front());
	}

	// Rasterizer components to descriptor set
	std::map <const Rasterizer *, vk::raii::DescriptorSet>
					_ds_components;
public:
	// Default constructor
	Raster() = default;

	// Constructors
	Raster(const Context &ctx, const vk::AttachmentLoadOp &load)
			: _ctx(ctx) {
		// Create render pass
		_render_pass = make_render_pass(ctx.device,
			ctx.swapchain_format,
			ctx.depth_format, load
		);

		// Create descriptor set layout
		_dsl = make_descriptor_set_layout(_ctx.device, _dsl_bindings);

		// Load all shaders
		auto shaders = make_shader_modules(_ctx.device, {
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/color_frag.spv",
			"shaders/bin/raster/normal_frag.spv",
			"shaders/bin/raster/blinn_phong_frag.spv"
		});

		// Push constants
		vk::PushConstantRange push_constants {
			vk::ShaderStageFlagBits::eVertex,
			0, sizeof(PushConstant)
		};

		// Pipeline layout
		_ppl = vk::raii::PipelineLayout(
			_ctx.device,
			{{}, *_dsl, push_constants}
		);

		// Pipeline cache
		vk::raii::PipelineCache pipeline_cache {
			_ctx.device,
			vk::PipelineCacheCreateInfo()
		};

		// Pipelines
		auto vertex_binding = Vertex::vertex_binding();
		auto vertex_attributes = Vertex::vertex_attributes();

		GraphicsPipelineInfo grp_info {
			.device = _ctx.device,
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
	}

	// Render
	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer,
			const ECS &ecs) {
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

					Device dev {
						_ctx.phdev,
						_ctx.device
					};

					// Update descriptor set
					rasterizer->bind_material(dev, _ds_components.at(rasterizer));
				}
			}
		}

		// Bind pipeline
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_p_normal);

		// Render all rasterizer components
		PushConstant push_constant {
			.view = camera.view(),
			.projection = camera.projection(),
			.albedo = glm::vec3 {1.0f, 0.0f, 1.0f},
			.type = Shading::eDiffuse,
			.highlight = false,
			.has_albedo = false,
			.has_normal = false
		};

		for (int i = 0; i < ecs.size(); i++) {
			if (!ecs.exists <Rasterizer> (i))
				continue;

			// Get transform
			Transform transform = ecs.get <Transform> (i);
			push_constant.model = transform.matrix();

			// Get rasterizer
			const Rasterizer *rasterizer = &ecs.get <Rasterizer> (i);
			const auto &ds = _ds_components.at(rasterizer);

			// Bind descriptor set
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*_ppl, 0, *ds, {}
			);

			// Bind vertex and index buffers
			rasterizer->bind_buffers(cmd);

			// Push constant
			cmd.pushConstants <PushConstant> (
				*_ppl, vk::ShaderStageFlagBits::eVertex,
				0, push_constant
			);

			// Draw
			cmd.drawIndexed(rasterizer->indices, 1, 0, 0, 0);
		}

		// End
		cmd.endRenderPass();
	}
};

}

}

#endif
