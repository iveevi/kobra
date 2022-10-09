// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>

// Engine headers
#include "../../include/camera.hpp"
#include "../../include/cuda/alloc.cuh"
#include "../../include/cuda/cast.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/cuda/interop.cuh"
#include "../../include/ecs.hpp"
#include "../../include/layers/hybrid_tracer.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/texture_manager.hpp"
#include "../../include/transform.hpp"
#include "../../shaders/raster/bindings.h"

// OptiX Source PTX
#define OPTIX_PTX_FILE "bin/ptx/hybrid_rt.ptx"

namespace kobra {

namespace layers {

// SBT record types
using RaygenRecord = optix::Record <int>;
using MissRecord = optix::Record <int>;
using HitRecord = optix::Record <optix::Hit>;

// Push constants
struct PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
};

// Static member variables
const std::vector <DSLB> HybridTracer::gbuffer_dsl_bindings {
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

const std::vector <DSLB> HybridTracer::present_dsl_bindings = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

// Allocate the framebuffer images
static void allocate_framebuffer_images(HybridTracer &layer, const Context &context, const vk::Extent2D &extent)
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

// Load OptiX program groups
static void load_optix_program_groups(HybridTracer &layer)
{
	static char log[2048];
	static size_t sizeof_log = sizeof(log);

	// Load programs
	OptixProgramGroupOptions program_options = {};

	OptixProgramGroupDesc program_desc1 {
		.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
		.raygen = {
			.module = layer.optix_module,
			.entryFunctionName = "__raygen__rg"
		}
	};

	OPTIX_CHECK_LOG(
		optixProgramGroupCreate(
			layer.optix_context,
			&program_desc1, 1,
			&program_options,
			log, &sizeof_log,
			&layer.optix_programs.raygen
		)
	);
	
	OptixProgramGroupDesc program_desc2 {
		.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
		.miss = {
			.module = layer.optix_module,
			.entryFunctionName = "__miss__ms"
		}
	};

	OPTIX_CHECK_LOG(
		optixProgramGroupCreate(
			layer.optix_context,
			&program_desc1, 1,
			&program_options,
			log, &sizeof_log,
			&layer.optix_programs.miss
		)
	);
	
	OptixProgramGroupDesc program_desc3 {
		.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
		.miss = {
			.module = layer.optix_module,
			.entryFunctionName = "__miss__shadow"
		}
	};

	OPTIX_CHECK_LOG(
		optixProgramGroupCreate(
			layer.optix_context,
			&program_desc3, 1,
			&program_options,
			log, &sizeof_log,
			&layer.optix_programs.shadow_miss
		)
	);
	
	OptixProgramGroupDesc program_desc4 {
		.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
		.hitgroup = {
			.moduleCH = layer.optix_module,
			.entryFunctionNameCH = "__closesthit__ch",
		}
	};

	OPTIX_CHECK_LOG(
		optixProgramGroupCreate(
			layer.optix_context,
			&program_desc4, 1,
			&program_options,
			log, &sizeof_log,
			&layer.optix_programs.hit
		)
	);
	
	OptixProgramGroupDesc program_desc5 {
		.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
		.hitgroup = {
			.moduleCH = layer.optix_module,
			.entryFunctionNameCH = "__closesthit__shadow",
		}
	};

	OPTIX_CHECK_LOG(
		optixProgramGroupCreate(
			layer.optix_context,
			&program_desc5, 1,
			&program_options,
			log, &sizeof_log,
			&layer.optix_programs.shadow_hit
		)
	);
}

// Create and configure OptiX pipeline
static void load_optix_pipeline
		(HybridTracer &layer,
		 const OptixPipelineCompileOptions &ppl_compile_options)
{
	static char log[2048];
	static size_t sizeof_log = sizeof(log);

	// Create the pipeline and configure it
	std::vector <OptixProgramGroup> program_groups {
		layer.optix_programs.raygen,
		layer.optix_programs.miss,
		layer.optix_programs.shadow_miss,
		layer.optix_programs.hit,
		layer.optix_programs.shadow_hit
	};

	OptixPipelineLinkOptions ppl_link_options = {};

	ppl_link_options.maxTraceDepth = 5;
	ppl_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		
	OPTIX_CHECK_LOG(
		optixPipelineCreate(
			layer.optix_context,
			&ppl_compile_options,
			&ppl_link_options,
			program_groups.data(),
			program_groups.size(),
			log, &sizeof_log,
			&layer.optix_pipeline
		)
	);

	// Set stack sizes
	OptixStackSizes stack_sizes = {};
	for (auto &program : program_groups) {
		OPTIX_CHECK(
			optixUtilAccumulateStackSizes(
				program, &stack_sizes
			)
		);
	}

	uint32_t direct_callable_stack_size_from_traversal = 0;
	uint32_t direct_callable_stack_size_from_state = 0;
	uint32_t continuation_stack_size = 0;

	OPTIX_CHECK(
		optixUtilComputeStackSizes(
			&stack_sizes,
			ppl_link_options.maxTraceDepth,
			0, 0,
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state,
			&continuation_stack_size
		)
	);

	OPTIX_CHECK(
		optixPipelineSetStackSize(
			layer.optix_pipeline,
			direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state,
			continuation_stack_size,
			2
		)
	);
}

// Setup and load OptiX things
static void initialize_optix(HybridTracer &layer)
{
	// Create the context
	layer.optix_context = optix::make_context();

	// Allocate a stream for the layer
	CUDA_CHECK(cudaStreamCreate(&layer.optix_stream));
	
	// Pipeline configuration
	OptixPipelineCompileOptions ppl_compile_options = {};

	ppl_compile_options.usesMotionBlur = false;
	ppl_compile_options.numPayloadValues = 2;
	ppl_compile_options.numAttributeValues = 0;
	ppl_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	ppl_compile_options.pipelineLaunchParamsVariableName = "ht_params";
	
	ppl_compile_options.usesPrimitiveTypeFlags =
		OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
	
	ppl_compile_options.traversableGraphFlags =
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

	// Module configuration
	OptixModuleCompileOptions module_options = {};

	module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

	// Now load the module
	char log[2048];
	size_t sizeof_log = sizeof(log);

	std::string file = common::read_file(OPTIX_PTX_FILE);
	OPTIX_CHECK_LOG(
		optixModuleCreateFromPTX(
			layer.optix_context,
			&module_options, &ppl_compile_options,
			file.c_str(), file.size(),
			log, &sizeof_log,
			&layer.optix_module
		)
	);

	// Load programs
	load_optix_program_groups(layer);

	// Create the pipeline and configure it
	load_optix_pipeline(layer, ppl_compile_options);

	// Create the shader binding table
	std::vector <RaygenRecord> rg_records(1);
	std::vector <MissRecord> ms_records(2);

	optix::pack_header(layer.optix_programs.raygen, rg_records[0]);

	optix::pack_header(layer.optix_programs.miss, ms_records[0]);
	optix::pack_header(layer.optix_programs.shadow_miss, ms_records[1]);

	CUdeviceptr d_raygen_sbt = cuda::make_buffer_ptr(rg_records);
	CUdeviceptr d_miss_sbt = cuda::make_buffer_ptr(ms_records);

	layer.optix_sbt.raygenRecord = d_raygen_sbt;
	
	layer.optix_sbt.missRecordBase = d_miss_sbt;
	layer.optix_sbt.missRecordCount = ms_records.size();
	layer.optix_sbt.missRecordStrideInBytes = sizeof(MissRecord);

	// Configure launch parameters
	layer.launch_params.resolution = {
		layer.extent.width,
		layer.extent.height
	};

	layer.launch_params.color_buffer = (float4 *)
		cuda::alloc(
			layer.extent.width * layer.extent.height
			* sizeof(float4)
		);

	// G-buffer results
	layer.launch_params.positions = layer.cuda_tex.positions;
	layer.launch_params.normals = layer.cuda_tex.normals;
	
	layer.launch_params.albedo = layer.cuda_tex.albedo;
	layer.launch_params.specular = layer.cuda_tex.specular;
	layer.launch_params.extra = layer.cuda_tex.extra;

	// Lights (set to null, etc)
	layer.launch_params.lights.quad_count = 0;
	layer.launch_params.lights.triangle_count = 0;

	// Allocate the parameters buffer
	layer.launch_params_buffer = cuda::alloc(
		sizeof(optix::HT_Parameters)
	);

	// Allocate truncated color buffer
	layer.truncated = cuda::alloc(
		layer.extent.width * layer.extent.height
		* sizeof(uint32_t)
	);

	// Start the timer
	layer.timer.start();
}

// Create the layer
// TOOD: all custom extent...
HybridTracer HybridTracer::make(const Context &context)
{
	// To return
	HybridTracer layer;

	// Extract critical Vulkan structures
	layer.device = context.device;
	layer.phdev = context.phdev;
	layer.descriptor_pool = context.descriptor_pool;

	layer.cmd = make_command_buffer(*context.device, *context.command_pool);

	// TODO: queue allocation system
	layer.queue = vk::raii::Queue {
		*context.device,
		0, 1
	};

	// Create the framebuffers
	layer.extent = context.extent;

	allocate_framebuffer_images(layer, context, layer.extent);

	layer.depth = DepthBuffer {
		*context.phdev, *context.device,
		vk::Format::eD32Sfloat, context.extent
	};

	// Export to CUDA
	layer.cuda_tex.positions = cuda::import_vulkan_texture_32f(*layer.device, layer.positions);
	layer.cuda_tex.normals = cuda::import_vulkan_texture_32f(*layer.device, layer.normals);
	layer.cuda_tex.albedo = cuda::import_vulkan_texture_32f(*layer.device, layer.albedo);
	layer.cuda_tex.specular = cuda::import_vulkan_texture_32f(*layer.device, layer.specular);
	layer.cuda_tex.extra = cuda::import_vulkan_texture_32f(*layer.device, layer.extra);

	// Initialize OptiX
	initialize_optix(layer);

	// Create the G-buffer generation render pass
	auto eClear = vk::AttachmentLoadOp::eClear;

	layer.gbuffer_render_pass = make_render_pass(*context.device,
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
		{}, *layer.gbuffer_render_pass,
		(uint32_t) attachments.size(),
		attachments.data(),
		context.extent.width,
		context.extent.height,
		1
	};

	layer.framebuffer = Framebuffer {*context.device, fb_info};

	// Create the present render pass
	layer.present_render_pass = make_render_pass(*context.device,
		{context.swapchain_format},
		{eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Descriptor set layout
	layer.gbuffer_dsl = make_descriptor_set_layout(*context.device, gbuffer_dsl_bindings);
	layer.present_dsl = make_descriptor_set_layout(*context.device, present_dsl_bindings);

	// Allocate present descriptor set
	auto dsets = vk::raii::DescriptorSets {
		*context.device,
		{**context.descriptor_pool, *layer.present_dsl}
	};

	layer.present_dset = std::move(dsets.front());

	// Push constants and pipeline layout
	vk::PushConstantRange gbuffer_push_constants {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	layer.gbuffer_ppl = PipelineLayout {
		*context.device,
		{{}, *layer.gbuffer_dsl, gbuffer_push_constants}
	};

	layer.present_ppl = PipelineLayout {
		*context.device,
		{{}, *layer.present_dsl, {}}
	};

	// Create the G-buffer generation pipeline
	auto shaders = make_shader_modules(*context.device, {
		"bin/spv/hybrid_deferred_vert.spv",
		"bin/spv/hybrid_deferred_frag.spv"
	});

	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo gbuffer_grp_info {
		*context.device, layer.gbuffer_render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		vertex_binding, vertex_attributes,
		layer.gbuffer_ppl
	};

	gbuffer_grp_info.color_blend_attachments = 5;
	gbuffer_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

	layer.gbuffer_pipeline = make_graphics_pipeline(gbuffer_grp_info);

	// Create the present pipeline
	shaders = make_shader_modules(*context.device, {
		"bin/spv/spit_vert.spv",
		"bin/spv/spit_frag.spv"
	});
	
	GraphicsPipelineInfo present_grp_info {
		*context.device, layer.present_render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		{}, {},
		layer.present_ppl
	};

	present_grp_info.no_bindings = true;
	present_grp_info.depth_test = false;
	present_grp_info.depth_write = false;

	layer.present_pipeline = make_graphics_pipeline(present_grp_info);

	// Allocate resources for rendering results

	// TODO: shared resource as a CUDA texture?
	layer.result_image = ImageData(
		*context.phdev, *context.device,
		vk::Format::eR8G8B8A8Unorm,
		context.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		vk::ImageLayout::eUndefined,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	layer.result_sampler = make_sampler(*context.device, layer.result_image);

	// Allocate staging buffer
	vk::DeviceSize stage_size = context.extent.width
		* context.extent.height
		* sizeof(uint32_t);

	auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
	auto mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal
		| vk::MemoryPropertyFlagBits::eHostCoherent
		| vk::MemoryPropertyFlagBits::eHostVisible;

	layer.result_buffer = BufferData(
		*context.phdev, *context.device, stage_size,
		usage | vk::BufferUsageFlagBits::eTransferSrc, mem_props
	);

	// Bind image sampler to the present descriptor set
	//	immediately, since it will not change
	bind_ds(*context.device,
		layer.present_dset,
		layer.result_sampler,
		layer.result_image, 0
	);

	// Return
	return layer;
}

// Create a descriptor set for the layer
static HybridTracer::RasterizerDset serve_dset(HybridTracer &layer, uint32_t count)
{
	std::vector <vk::DescriptorSetLayout> layouts(count, *layer.gbuffer_dsl);

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
static void configure_dset(HybridTracer &layer,
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

// Update the light buffers if needed
static void update_light_buffers(HybridTracer &layer,
		const std::vector <const Light *> &lights,
		const std::vector <const Transform *> &light_transforms)
{
	if (layer.host.quad_lights.size() != lights.size()) {
		layer.host.quad_lights.resize(lights.size());
	
		auto &quad_lights = layer.host.quad_lights;
		for (int i = 0; i < lights.size(); i++) {
			const Light *light = lights[i];
			const Transform *transform = light_transforms[i];
			
			glm::vec3 a {-0.5f, 0, -0.5f};
			glm::vec3 b {0.5f, 0, -0.5f};
			glm::vec3 c {-0.5f, 0, 0.5f};

			a = transform->apply(a);
			b = transform->apply(b);
			c = transform->apply(c);

			quad_lights[i].a = cuda::to_f3(a);
			quad_lights[i].ab = cuda::to_f3(b - a);
			quad_lights[i].ac = cuda::to_f3(c - a);
			quad_lights[i].intensity = cuda::to_f3(light->power * light->color);
		}

		layer.launch_params.lights.quads = cuda::make_buffer(quad_lights);
		layer.launch_params.lights.quad_count = quad_lights.size();
	}
}

// Build or update acceleration structures for the scene
static void update_acceleration_structure(HybridTracer &layer,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	int submesh_count = submeshes.size();

	// Build acceleration structures
	OptixAccelBuildOptions gas_accel_options = {};
	gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	gas_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	std::vector <OptixTraversableHandle> instance_gas(submesh_count);
	
	// Flags
	const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

	KOBRA_LOG_FUNC(Log::INFO) << "Building GAS for instances (# = "
		<< submesh_count << ")" << std::endl;

	// TODO: CACHE the vertices for the sbts
	// TODO: reuse the vertex buffer from the rasterizer

	for (int i = 0; i < submesh_count; i++) {
		const Submesh *s = submeshes[i];

		// Prepare submesh vertices and triangles
		std::vector <float3> vertices;
		std::vector <uint3> triangles;
		
		// TODO: method to generate accel handle from cuda buffers
		for (int j = 0; j < s->indices.size(); j += 3) {
			triangles.push_back({
				s->indices[j],
				s->indices[j + 1],
				s->indices[j + 2]
			});
		}

		for (int j = 0; j < s->vertices.size(); j++) {
			auto p = s->vertices[j].position;
			vertices.push_back(cuda::to_f3(p));
		}

		// Create the build input
		OptixBuildInput build_input {};

		build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		CUdeviceptr d_vertices = cuda::make_buffer_ptr(vertices);
		CUdeviceptr d_triangles = cuda::make_buffer_ptr(triangles);

		OptixBuildInputTriangleArray &triangle_array = build_input.triangleArray;
		triangle_array.vertexFormat	= OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_array.numVertices	= vertices.size();
		triangle_array.vertexBuffers	= &d_vertices;

		triangle_array.indexFormat	= OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangle_array.numIndexTriplets	= triangles.size();
		triangle_array.indexBuffer	= d_triangles;

		triangle_array.flags		= triangle_input_flags;

		// SBT record properties
		triangle_array.numSbtRecords	= 1;
		triangle_array.sbtIndexOffsetBuffer = 0;
		triangle_array.sbtIndexOffsetStrideInBytes = 0;
		triangle_array.sbtIndexOffsetSizeInBytes = 0;

		// Build GAS
		CUdeviceptr d_gas_output;
		CUdeviceptr d_gas_tmp;

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(
			optixAccelComputeMemoryUsage(
				layer.optix_context, &gas_accel_options,
				&build_input, 1,
				&gas_buffer_sizes
			)
		);
		
		KOBRA_LOG_FUNC(Log::INFO) << "GAS buffer sizes: " << gas_buffer_sizes.tempSizeInBytes
			<< " " << gas_buffer_sizes.outputSizeInBytes << std::endl;

		d_gas_output = cuda::alloc(gas_buffer_sizes.outputSizeInBytes);
		d_gas_tmp = cuda::alloc(gas_buffer_sizes.tempSizeInBytes);

		OptixTraversableHandle handle;
		OPTIX_CHECK(
			optixAccelBuild(layer.optix_context,
				0, &gas_accel_options,
				&build_input, 1,
				d_gas_tmp, gas_buffer_sizes.tempSizeInBytes,
				d_gas_output, gas_buffer_sizes.outputSizeInBytes,
				&handle, nullptr, 0
			)
		);

		instance_gas[i] = handle;

		// Free data at the end
		cuda::free(d_vertices);
		cuda::free(d_triangles);
		cuda::free(d_gas_tmp);
	}

	// Build instances and top level acceleration structure
	std::vector <OptixInstance> instances;

	for (int i = 0; i < submesh_count; i++) {
		glm::mat4 mat = submesh_transforms[i]->matrix();

		float transform[12] = {
			mat[0][0], mat[1][0], mat[2][0], mat[3][0],
			mat[0][1], mat[1][1], mat[2][1], mat[3][1],
			mat[0][2], mat[1][2], mat[2][2], mat[3][2]
		};

		OptixInstance instance {};
		memcpy(instance.transform, transform, sizeof(float) * 12);

		// Set the instance handle
		instance.traversableHandle = instance_gas[i];
		instance.visibilityMask = 0b1;
		instance.sbtOffset = i;

		instances.push_back(instance);
	}

	// Create top level acceleration structure
	CUdeviceptr d_instances = cuda::make_buffer_ptr(instances);

	// TLAS for objects and lights
	{
		OptixBuildInput ias_build_input {};
		ias_build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		ias_build_input.instanceArray.instances = d_instances;
		ias_build_input.instanceArray.numInstances = instances.size();

		// IAS options
		OptixAccelBuildOptions ias_accel_options {};
		ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// IAS buffer sizes
		OptixAccelBufferSizes ias_buffer_sizes;
		OPTIX_CHECK(
			optixAccelComputeMemoryUsage(
				layer.optix_context, &ias_accel_options,
				&ias_build_input, 1,
				&ias_buffer_sizes
			)
		);

		KOBRA_LOG_FUNC(Log::INFO) << "IAS buffer sizes: " << ias_buffer_sizes.tempSizeInBytes << " " << ias_buffer_sizes.outputSizeInBytes << std::endl;

		// Allocate the IAS
		CUdeviceptr d_ias_output = cuda::alloc(ias_buffer_sizes.outputSizeInBytes);
		CUdeviceptr d_ias_tmp = cuda::alloc(ias_buffer_sizes.tempSizeInBytes);

		// Build the IAS
		OPTIX_CHECK(
			optixAccelBuild(layer.optix_context,
				0, &ias_accel_options,
				&ias_build_input, 1,
				d_ias_tmp, ias_buffer_sizes.tempSizeInBytes,
				d_ias_output, ias_buffer_sizes.outputSizeInBytes,
				&layer.optix.handle, nullptr, 0
			)
		);

		cuda::free(d_ias_tmp);
		cuda::free(d_instances);
	}

	// Copy address to launch parameters
	layer.launch_params.traversable = layer.optix.handle;
}

static void generate_submesh_data
		(const Submesh &submesh,
		const Transform &transform,
		optix::Hit &data)
{
	std::vector <float3> vertices(submesh.vertices.size());
	std::vector <float2> uvs(submesh.vertices.size());
	std::vector <uint3> triangles(submesh.triangles());
	
	std::vector <float3> normals(submesh.vertices.size());
	std::vector <float3> tangents(submesh.vertices.size());
	std::vector <float3> bitangents(submesh.vertices.size());

	int vertex_index = 0;
	int uv_index = 0;
	int triangle_index = 0;
	
	int normal_index = 0;
	int tangent_index = 0;
	int bitangent_index = 0;

	for (int j = 0; j < submesh.vertices.size(); j++) {
		glm::vec3 n = submesh.vertices[j].normal;
		glm::vec3 t = submesh.vertices[j].tangent;
		glm::vec3 b = submesh.vertices[j].bitangent;
		
		glm::vec3 v = submesh.vertices[j].position;
		glm::vec2 uv = submesh.vertices[j].tex_coords;

		v = transform.apply(v);
		n = transform.apply_vector(n);
		t = transform.apply_vector(t);
		b = transform.apply_vector(b);
		
		normals[normal_index++] = {n.x, n.y, n.z};
		tangents[tangent_index++] = {t.x, t.y, t.z};
		bitangents[bitangent_index++] = {b.x, b.y, b.z};

		vertices[vertex_index++] = {v.x, v.y, v.z};
		uvs[uv_index++] = {uv.x, uv.y};
	}

	for (int j = 0; j < submesh.indices.size(); j += 3) {
		triangles[triangle_index++] = {
			submesh.indices[j],
			submesh.indices[j + 1],
			submesh.indices[j + 2]
		};
	}

	// Store the data
	// TODO: cache later
	data.vertices = cuda::make_buffer(vertices);
	data.texcoords = cuda::make_buffer(uvs);

	data.normals = cuda::make_buffer(normals);
	data.tangents = cuda::make_buffer(tangents);
	data.bitangents = cuda::make_buffer(bitangents);

	data.triangles = cuda::make_buffer(triangles);
}

// Update the SBT data
static void update_sbt_data(HybridTracer &layer,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	int submesh_count = submeshes.size();

	std::vector <HitRecord> hit_records(submesh_count);
	for (int i = 0; i < submesh_count; i++) {
		const Submesh *submesh = submeshes[i];

		// Material
		Material mat = submesh->material;

		cuda::Material material;
		material.diffuse = cuda::to_f3(mat.diffuse);
		material.specular = cuda::to_f3(mat.specular);
		material.emission = cuda::to_f3(mat.emission);
		material.ambient = cuda::to_f3(mat.ambient);
		material.shininess = mat.shininess;
		material.roughness = mat.roughness;
		material.refraction = mat.refraction;
		material.type = mat.type;

		HitRecord hit_record {};
		hit_record.data.material = material;

		generate_submesh_data(*submesh, *submesh_transforms[i], hit_record.data);

		/* Import textures if necessary
		// TODO: method?
		if (mat.has_albedo()) {
			const ImageData &diffuse = TextureManager::load_texture(
				_ctx.dev(), mat.albedo_texture
			);

			hg_sbt.data.textures.diffuse = cuda::import_vulkan_texture(*_ctx.device, diffuse);
			hg_sbt.data.textures.has_diffuse = true;
		}

		if (mat.has_normal()) {
			const ImageData &normal = TextureManager::load_texture(
				_ctx.dev(), mat.normal_texture
			);

			hg_sbt.data.textures.normal = cuda::import_vulkan_texture(*_ctx.device, normal);
			hg_sbt.data.textures.has_normal = true;
		}

		if (mat.has_roughness()) {
			const ImageData &roughness = TextureManager::load_texture(
				_ctx.dev(), mat.roughness_texture
			);

			hg_sbt.data.textures.roughness = cuda::import_vulkan_texture(*_ctx.device, roughness);
			hg_sbt.data.textures.has_roughness = true;
		} */
	
		optix::pack_header(layer.optix_programs.hit, hit_record);
		hit_records[i] = hit_record;
	}
	
	// TODO: delete hit shadow below

	// Duplicate the SBTs for the shadow program
	int size = hit_records.size();

	hit_records.resize(size * 2);
	for (int i = 0; i < size; i++) {
		HitRecord hit_record = hit_records[i];
		optix::pack_header(layer.optix_programs.shadow_hit, hit_record);
		hit_records[size + i] = hit_record;
	}

	// Update the SBT
	CUdeviceptr d_hit_records = cuda::make_buffer_ptr(hit_records);
	
	layer.optix_sbt.hitgroupRecordBase = d_hit_records;
	layer.optix_sbt.hitgroupRecordCount = hit_records.size();
	layer.optix_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);

	layer.launch_params.instances = submesh_count;
}

// Render the deferred stage (generate the G-buffer)

// TODO: perform this in a separate command buffer than the main one used to
// present, etc (and separate queue)
static void generate_gbuffers(HybridTracer &layer,
		const CommandBuffer &cmd,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{

	// Preprocess the entities
	std::vector <const Rasterizer *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// TODO: one unifying renderer component, with options for
		// raytracing, etc
		if (ecs.exists <Rasterizer> (i)) {
			const auto *rasterizer = &ecs.get <Rasterizer> (i);
			const auto *transform = &ecs.get <Transform> (i);

			rasterizers.push_back(rasterizer);
			rasterizer_transforms.push_back(transform);

			// If not it the dsets dictionary, create it
			if (layer.gbuffer_dsets.find(rasterizer) ==
					layer.gbuffer_dsets.end()) {
				layer.gbuffer_dsets[rasterizer] = serve_dset(
					layer,
					rasterizer->materials.size()
				);

				// Configure the dset
				configure_dset(layer, layer.gbuffer_dsets[rasterizer], rasterizer);
			}
		}
		
		if (ecs.exists <Light> (i)) {
			const auto *light = &ecs.get <Light> (i);
			const auto *transform = &ecs.get <Transform> (i);

			lights.push_back(light);
			light_transforms.push_back(transform);
		}
	}

	// Check if an update is needed
	bool update = false;
	if (rasterizers.size() == layer.cache.rasterizers.size()) {
		for (int i = 0; i < rasterizers.size(); i++) {
			if (rasterizers[i] != layer.cache.rasterizers[i]) {
				update = true;
				break;
			}
		}
	} else {
		update = true;
	}

	// Update data if necessary 
	if (update) {
		// Update the cache
		layer.cache.rasterizers = rasterizers;
	
		// Load the list of all submeshes
		std::vector <const Submesh *> submeshes;
		std::vector <const Transform *> submesh_transforms;

		for (int i = 0; i < rasterizers.size(); i++) {
			const Rasterizer *rasterizer = rasterizers[i];
			const Transform *transform = rasterizer_transforms[i];

			for (int j = 0; j < rasterizer->mesh->submeshes.size(); j++) {
				const Submesh *submesh = &rasterizer->mesh->submeshes[j];

				submeshes.push_back(submesh);
				submesh_transforms.push_back(transform);
			}
		}

		// Update the data
		update_light_buffers(layer, lights, light_transforms);
		update_acceleration_structure(layer, submeshes, submesh_transforms);
		update_sbt_data(layer, submeshes, submesh_transforms);
	}

	// Set viewing position
	layer.launch_params.camera = cuda::to_f3(transform.position);

	// Get time
	layer.launch_params.time = layer.timer.elapsed_start();

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
			*layer.gbuffer_render_pass,
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
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *layer.gbuffer_pipeline);

	// Setup push constants
	PushConstants pc;

	pc.view = camera.view_matrix(transform);
	pc.projection = camera.perspective_matrix();

	// Render all entities with a rasterizer component
	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <Rasterizer> (i)) {
			pc.model = ecs.get <Transform> (i).matrix();

			cmd.pushConstants <PushConstants> (*layer.gbuffer_ppl,
				vk::ShaderStageFlagBits::eVertex,
				0, pc
			);

			// Bind and draw
			const Rasterizer &rasterizer = ecs.get <Rasterizer> (i);
			const HybridTracer::RasterizerDset &dset
				= layer.gbuffer_dsets[&rasterizer];

			int submeshes = rasterizer.size();
			for (int i = 0; i < submeshes; i++) {
				// Bind buffers
				cmd.bindVertexBuffers(0, *rasterizer.get_vertex_buffer(i).buffer, {0});
				cmd.bindIndexBuffer(*rasterizer.get_index_buffer(i).buffer,
					0, vk::IndexType::eUint32
				);

				// Bind descriptor set
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
					*layer.gbuffer_ppl, 0, *dset[i], {}
				);

				// Draw
				cmd.drawIndexed(rasterizer.get_index_count(i), 1, 0, 0, 0);
			}
		}
	}

	// End render pass
	cmd.endRenderPass();
}

// Tone maps: 0 for sRGB, 1 for ACES
// TODO: kernel.cuh
static __global__ void compute_pixel_values
		(float4 *pixels, uint32_t *target,
		int width, int height, int tonemapping = 0)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int in_idx = y * width + x;
	int out_idx = (height - y - 1) * width + x;

	float4 pixel = pixels[in_idx];
	uchar4 color = cuda::make_color(pixel, tonemapping);
	target[out_idx] = cuda::to_ui32(color);
}

// Path tracing computation
void compute(HybridTracer &layer,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{
	// Generate the G-buffer
	layer.cmd.begin({});
		generate_gbuffers(layer, layer.cmd, ecs, camera, transform);
	layer.cmd.end();

	// Submit the command buffer
	layer.queue.submit(
		vk::SubmitInfo {
			0, nullptr,
			nullptr,
			1, &*layer.cmd,
			0, nullptr
		},
		nullptr
	);

	// Wait for the queue to finish
	layer.queue.waitIdle();

	// Copy parameters to the GPU
	cuda::copy(
		layer.launch_params_buffer,
		&layer.launch_params, 1,
		cudaMemcpyHostToDevice
	);
	
	// TODO: depth?
	OPTIX_CHECK(
		optixLaunch(
			layer.optix_pipeline,
			layer.optix_stream,
			layer.launch_params_buffer,
			sizeof(optix::HT_Parameters),
			&layer.optix_sbt,
			layer.extent.width, layer.extent.height, 1
		)
	);
	
	CUDA_SYNC_CHECK();

	// Conversion kernel
	uint width = layer.extent.width;
	uint height = layer.extent.height;

	dim3 block(16, 16);
	dim3 grid(
		(width + block.x - 1)/block.x,
		(height + block.y - 1)/block.y
	);

	compute_pixel_values <<<grid, block>>> (
		layer.launch_params.color_buffer,
		(uint32_t *) layer.truncated,
		width, height, 1
	);
	
	// Copy results to the CPU
	uint32_t size = layer.extent.width * layer.extent.height;
	if (layer.color_buffer.size() != size)
		layer.color_buffer.resize(size);

	cuda::copy(
		layer.color_buffer,
		layer.truncated,
		size
	);
}

// Render to the presentable framebuffer
void render(HybridTracer &layer,
		const CommandBuffer &cmd,
		const Framebuffer &framebuffer,
		const RenderArea &ra)
{
	// Upload data to the buffer
	layer.result_buffer.upload(layer.color_buffer);
	
	// Copy buffer to image
	layer.result_image.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

	copy_data_to_image(cmd,
		layer.result_buffer.buffer,
		layer.result_image.image,
		layer.result_image.format,
		layer.extent.width, layer.extent.height
	);

	// Transition image back to shader read
	layer.result_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);	
	
	// Apply render area
	ra.apply(cmd, layer.extent);

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
			*layer.present_render_pass,
			*framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				layer.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Presentation pipeline
	cmd.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		*layer.present_pipeline
	);

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*layer.present_ppl, 0, {*layer.present_dset}, {}
	);

	// Draw and end
	cmd.draw(6, 1, 0, 0);
	cmd.endRenderPass();
}

}

}
