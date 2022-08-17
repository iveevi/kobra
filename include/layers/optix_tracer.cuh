#ifndef KOBRA_LAYERS_OPTIX_TRACER_H_
#define KOBRA_LAYERS_OPTIX_TRACER_H_

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "../backend.hpp"
#include "../cuda/buffer_data.cuh"
#include "../ecs.hpp"

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
	OptixDeviceContext		_optix_ctx = nullptr;
	OptixModule			_optix_module = nullptr;
	OptixPipeline			_optix_pipeline = nullptr;
	OptixShaderBindingTable		_optix_sbt;
	CUstream			_optix_stream;

	// OptiX acceleration structures
	struct {
		OptixTraversableHandle	pure_objects;
		OptixTraversableHandle	all_objects;
	} _traversables;

	// OptiX program groups
	struct {
		OptixProgramGroup	raygen = nullptr;
		OptixProgramGroup	hit_radiance = nullptr;
		OptixProgramGroup	hit_shadow = nullptr;
		OptixProgramGroup	miss_radiance = nullptr;
		OptixProgramGroup	miss_shadow = nullptr;
	} _programs;

	CUdeviceptr			_optix_hg_sbt = 0;
	CUdeviceptr			_optix_miss_sbt = 0;

	// Buffers
	struct {
		CUdeviceptr		area_lights;
	} _buffers;

	// Output resources
	cuda::BufferData		_result_buffer;
	std::vector <uint32_t>		_output;

	// Images
	// TODO: wrap in struct
	cudaExternalMemory_t		_dext_env_map = nullptr;
	CUdeviceptr			_d_environment_map = 0;
	cudaTextureObject_t		_tex_env_map = 0;
	const ImageData			*_v_environment_map = nullptr;

	// Cached entity data
	struct {
		std::vector <const Light *> lights;
		std::vector <const Transform *> light_transforms;
	} _cached;

	std::vector <const kobra::Raytracer *>
					_c_raytracers;
	std::vector <Transform>		_c_transforms;

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

		grp_info.no_bindings = true;

		_pipeline = make_graphics_pipeline(grp_info);
	}

	// Other helper methods
	void _optix_build();
	void _optix_update_materials();
	void _optix_trace(const Camera &, const Transform &);
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
			: _ctx(ctx), _result_buffer(width * height * 4) {
		_initialize_optix();
		_initialize_vulkan_structures(load);

		// Allocate image and sampler
		_result = ImageData(
			*_ctx.phdev, *_ctx.device,
			vk::Format::eR8G8B8A8Unorm, {width, height},
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageLayout::eUndefined,
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

	// Set environment map
	void environment_map(const std::string &);

	// Render
	void render(const vk::raii::CommandBuffer &,
			const vk::raii::Framebuffer &,
			const ECS &, const RenderArea & = {{-1, -1}, {-1, -1}});
};

}

}

#endif
