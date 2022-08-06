#ifndef KOBRA_LAYERS_OPTIX_TRACER_H_
#define KOBRA_LAYERS_OPTIX_TRACER_H_

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "../backend.hpp"

namespace kobra {

namespace layers {

class OptixTracer {
	// Vulkan context
	Context _ctx;

	// Other Vulkan structures
	vk::raii::RenderPass		_render_pass = nullptr;
	vk::raii::PipelineLayout	_ppl = nullptr;
	vk::raii::Pipeline		_pipeline = nullptr;
	
	// Descriptor set, layout and bindings
	vk::raii::DescriptorSet		_ds_render = nullptr;
	vk::raii::DescriptorSetLayout	_dsl_render = nullptr;
	static const std::vector <DSLB> _dslb_render;

	// OptiX structures
	OptixModule			_module = nullptr;

	std::vector <uint32_t>		_output;

	// Initialize Optix globally
	void _initialize_optix();

	struct Viewport {
		uint width;
		uint height;
	};

	// Initialize Vulkan resources
	void _initialize_vulkan_structures(const vk::AttachmentLoadOp &load) {
		// Render pass
		_render_pass = make_render_pass(*_ctx.device,
			_ctx.swapchain_format,
			_ctx.depth_format, load
		);

		// Create descriptor set layouts
		_dsl_render = make_descriptor_set_layout(*_ctx.device, _dslb_render);

		// Create descriptor sets
		std::array <vk::DescriptorSetLayout, 1> dsls {
			*_dsl_render
		};

		auto dsets = vk::raii::DescriptorSets {
			*_ctx.device,
			{**_ctx.descriptor_pool, dsls}
		};

		_ds_render = std::move(dsets.front());

		// Load all shaders
		auto shaders = make_shader_modules(*_ctx.device, {
			"shaders/bin/generic/postproc_vert.spv",
			"shaders/bin/generic/postproc_frag.spv"
		});

		// Postprocess pipeline
		// TODO: is this even needed?
		auto pcr = vk::PushConstantRange {
			vk::ShaderStageFlagBits::eVertex,
			0, sizeof(Viewport)
		};

		_ppl = vk::raii::PipelineLayout {
			*_ctx.device,
			{{}, *_dsl_render, pcr}
		};

		GraphicsPipelineInfo grp_info(*_ctx.device, _render_pass,
			std::move(shaders[0]), nullptr,
			std::move(shaders[1]), nullptr,
			{}, {},
			_ppl, vk::raii::PipelineCache {
				*_ctx.device, nullptr
			}
		);

		_pipeline = make_graphics_pipeline(grp_info);
	}
public:
	// Default constructor
	OptixTracer() = default;

	static constexpr unsigned int width = 1280;
	static constexpr unsigned int height = 720;

	// Resulting vulkan image
	ImageData _result = nullptr;
	vk::raii::Sampler _sampler = nullptr;

	// Staging buffer for OptiX output
	BufferData _staging = nullptr;

	// Constructor
	OptixTracer(const Context &ctx, const vk::AttachmentLoadOp &load)
			: _ctx(ctx) {
		_initialize_optix();
		_initialize_vulkan_structures(load);

		// Allocate image and sampler
		_result = ImageData(
			*_ctx.phdev, *_ctx.device,
			vk::Format::eR8G8B8A8Unorm, {width, height},
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageLayout::ePreinitialized,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor
		);

		_sampler = make_sampler(*_ctx.device, _result);

		// Allocate staging buffer
		vk::DeviceSize stage_size = width * height * sizeof(uint32_t);

		auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
		auto mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal
			| vk::MemoryPropertyFlagBits::eHostCoherent
			| vk::MemoryPropertyFlagBits::eHostVisible;

		_staging = BufferData(
			*_ctx.phdev, *_ctx.device, stage_size,
			usage | vk::BufferUsageFlagBits::eTransferSrc, mem_props
		);
	
		// Bind sampler
		bind_ds(*_ctx.device,
			_ds_render,
			_sampler,
			_result, 0
		);
	}

	// Render
	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer,
			const RenderArea &ra = {{-1, -1}, {-1, -1}}) {
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

		// Copy output to staging buffer
		_staging.upload(_output);

		// Copy staging buffer to image
		_result.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

		copy_data_to_image(cmd,
			_staging.buffer,
			_result.image,
			_result.format,
			width, height
		);

		// Transition image back to shader read
		_result.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

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

		// Post process pipeline
		cmd.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			*_pipeline
		);

		// Bind descriptor set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*_ppl, 0, {*_ds_render}, {}
		);

		// Draw and end
		cmd.draw(6, 1, 0, 0);
		cmd.endRenderPass();
	}
};

}

}

#endif
