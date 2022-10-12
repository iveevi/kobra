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
#include "../../include/layers/wadjet.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/texture_manager.hpp"
#include "../../include/transform.hpp"
#include "../../shaders/raster/bindings.h"
#include "../../include/profiler.hpp"

// OptiX Source PTX
#define OPTIX_PTX_FILE "bin/ptx/wadjet_rt.ptx"

namespace kobra {

namespace layers {

// SBT record types
using RaygenRecord = optix::Record <int>;
using MissRecord = optix::Record <int>;
using HitRecord = optix::Record <optix::Hit>;

// Static member variables
const std::vector <DSLB> Wadjet::dsl_bindings = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

// Load OptiX program groups
static void load_optix_program_groups(Wadjet &layer)
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
			&program_desc2, 1,
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
	
	// TODO: get rid of shadow hit
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
		(Wadjet &layer,
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

	ppl_link_options.maxTraceDepth = 10;
	ppl_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		
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

	KOBRA_LOG_FUNC(Log::INFO) << "OptiX pipeline created: "
		<< "direct traversable = " << direct_callable_stack_size_from_traversal << ", "
		<< "direct state = " << direct_callable_stack_size_from_state << ", "
		<< "continuation = " << continuation_stack_size << std::endl;
}

// Setup and load OptiX things
static void initialize_optix(Wadjet &layer)
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

	ppl_compile_options.pipelineLaunchParamsVariableName = "parameters";
	
	ppl_compile_options.usesPrimitiveTypeFlags =
		OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
	
	ppl_compile_options.traversableGraphFlags =
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

	// Module configuration
	OptixModuleCompileOptions module_options = {};

	module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

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
	auto &params = layer.launch_params;

	int width = layer.extent.width;
	int height = layer.extent.height;

	params.resolution = {
		layer.extent.width,
		layer.extent.height
	};

	// Lights (set to null, etc)
	layer.launch_params.lights.quad_count = 0;
	layer.launch_params.lights.triangle_count = 0;

	// Accumulatoin on by default
	layer.launch_params.accumulate = true;

	// Advanced sampling resources
	float radius = std::min(width, height)/10.0f;

	std::vector <optix::ReSTIR_Reservoir> r_temporal(width * height, 30);
	std::vector <optix::ReSTIR_Reservoir> r_spatial(width * height, 200);
	std::vector <float> sampling_radii(width * height, radius);

	params.advanced.r_temporal = cuda::make_buffer(r_temporal);
	params.advanced.r_temporal_prev = cuda::make_buffer(r_temporal);
	
	params.advanced.r_spatial = cuda::make_buffer(r_spatial);
	params.advanced.r_spatial_prev = cuda::make_buffer(r_spatial);

	params.advanced.sampling_radii = cuda::make_buffer(sampling_radii);

	// Color buffer (output)
	layer.launch_params.color_buffer = (float4 *)
		cuda::alloc(
			layer.extent.width * layer.extent.height
			* sizeof(float4)
		);

	std::cout << "Color buffer of size " << layer.extent.width << "x" << layer.extent.height << std::endl;

	// Allocate the parameters buffer
	layer.launch_params_buffer = cuda::alloc(
		sizeof(optix::WadjetParameters)
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
Wadjet Wadjet::make(const Context &context)
{
	// To return
	Wadjet layer;

	// Extract critical Vulkan structures
	layer.device = context.device;
	layer.phdev = context.phdev;
	layer.descriptor_pool = context.descriptor_pool;

	// Create the framebuffers
	layer.extent = context.extent;

	layer.depth = DepthBuffer {
		*context.phdev, *context.device,
		vk::Format::eD32Sfloat, context.extent
	};

	// Initialize OptiX
	initialize_optix(layer);

	// Create the present render pass
	layer.render_pass = make_render_pass(*context.device,
		{context.swapchain_format},
		{vk::AttachmentLoadOp::eClear},
		context.depth_format,
		vk::AttachmentLoadOp::eClear
	);

	// Descriptor set layout
	layer.dsl = make_descriptor_set_layout(*context.device, dsl_bindings);

	// Allocate present descriptor set
	auto dsets = vk::raii::DescriptorSets {
		*context.device,
		{**context.descriptor_pool, *layer.dsl}
	};

	layer.dset = std::move(dsets.front());

	// Push constants and pipeline layout
	layer.ppl = PipelineLayout {
		*context.device,
		{{}, *layer.dsl, {}}
	};

	// Create the present pipeline
	auto shaders = make_shader_modules(*context.device, {
		"bin/spv/spit_vert.spv",
		"bin/spv/spit_frag.spv"
	});
	
	GraphicsPipelineInfo present_grp_info {
		*context.device, layer.render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		{}, {},
		layer.ppl
	};

	present_grp_info.no_bindings = true;
	present_grp_info.depth_test = false;
	present_grp_info.depth_write = false;

	layer.pipeline = make_graphics_pipeline(present_grp_info);

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
		layer.dset,
		layer.result_sampler,
		layer.result_image, 0
	);

	// Return
	return layer;
}

// Set the environment map
void set_envmap(Wadjet &layer, const std::string &path)
{
	// First load the environment map
	const auto &map = TextureManager::load_texture(
		*layer.phdev, *layer.device, path, true
	);

	layer.launch_params.envmap = cuda::import_vulkan_texture(*layer.device, map);
}

// Capture frame data
void capture(Wadjet &layer, std::vector <uint8_t> &data)
{
	int width = layer.extent.width;
	int height = layer.extent.height;

	// Copy the result image to the staging buffer
	if (data.size() != width * height * 4)
		data.resize(width * height * 4);

	std::memcpy(data.data(), layer.color_buffer.data(), data.size());
}

// Update the light buffers if needed
static void update_light_buffers(Wadjet &layer,
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
static void update_acceleration_structure(Wadjet &layer,
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

	KOBRA_LOG_FUNC(Log::INFO) << "Building GAS for instances (# = " << submesh_count << ")" << std::endl;

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
static void update_sbt_data(Wadjet &layer,
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

		// Import textures if necessary
		// TODO: method?
		if (mat.has_albedo()) {
			const ImageData &diffuse = TextureManager::load_texture(
				*layer.phdev, *layer.device, mat.albedo_texture
			);

			hit_record.data.textures.diffuse
				= cuda::import_vulkan_texture(*layer.device, diffuse);
			hit_record.data.textures.has_diffuse = true;
		}

		if (mat.has_normal()) {
			const ImageData &normal = TextureManager::load_texture(
				*layer.phdev, *layer.device, mat.normal_texture
			);

			hit_record.data.textures.normal
				= cuda::import_vulkan_texture(*layer.device, normal);
			hit_record.data.textures.has_normal = true;
		}

		if (mat.has_roughness()) {
			const ImageData &roughness = TextureManager::load_texture(
				*layer.phdev, *layer.device, mat.roughness_texture
			);

			hit_record.data.textures.roughness
				= cuda::import_vulkan_texture(*layer.device, roughness);
			hit_record.data.textures.has_roughness = true;
		}
	
		optix::pack_header(layer.optix_programs.hit, hit_record);
		hit_records[i] = hit_record;
	}
	
	// TODO: delete hit shadow below

	// Duplicate the SBTs for the shadow program
	// TODO: just get rid of this altogether
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

// Preprocess scene data

// TODO: perform this in a separate command buffer than the main one used to
// present, etc (and separate queue)
static void preprocess_scene(Wadjet &layer,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{

	// Preprocess the entities
	std::vector <const Rasterizer *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	{
		KOBRA_PROFILE_TASK(Preprocess of ECS)

		for (int i = 0; i < ecs.size(); i++) {
			// TODO: one unifying renderer component, with options for
			// raytracing, etc
			if (ecs.exists <Rasterizer> (i)) {
				const auto *rasterizer = &ecs.get <Rasterizer> (i);
				const auto *transform = &ecs.get <Transform> (i);

				rasterizers.push_back(rasterizer);
				rasterizer_transforms.push_back(transform);
			}
			
			if (ecs.exists <Light> (i)) {
				const auto *light = &ecs.get <Light> (i);
				const auto *transform = &ecs.get <Transform> (i);

				lights.push_back(light);
				light_transforms.push_back(transform);
			}
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
		KOBRA_PROFILE_TASK(Actually perform the update)

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

		// Reset the number of samples stored
		layer.launch_params.samples = 0;
	}

	// Set viewing position
	layer.launch_params.camera = cuda::to_f3(transform.position);
	
	auto uvw = kobra::uvw_frame(camera, transform);

	layer.launch_params.cam_u = cuda::to_f3(uvw.u);
	layer.launch_params.cam_v = cuda::to_f3(uvw.v);
	layer.launch_params.cam_w = cuda::to_f3(uvw.w);

	// Get time
	layer.launch_params.time = layer.timer.elapsed_start();
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
void compute(Wadjet &layer,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform,
		bool accumulate)
{
	KOBRA_PROFILE_TASK(HyrbidTracer compute path tracing);

	// Preprocess the scene
	{
		KOBRA_PROFILE_TASK(Update data);

		preprocess_scene(layer, ecs, camera, transform);
	}

	// Reset the accumulation state if needed
	if (!accumulate)
		layer.launch_params.samples = 0;

	// Copy parameters to the GPU
	cuda::copy(
		layer.launch_params_buffer,
		&layer.launch_params, 1,
		cudaMemcpyHostToDevice
	);
	
	{
		KOBRA_PROFILE_TASK(OptiX path tracing);
		
		int width = layer.extent.width;
		int height = layer.extent.height;

		// TODO: depth?
		OPTIX_CHECK(
			optixLaunch(
				layer.optix_pipeline,
				layer.optix_stream,
				layer.launch_params_buffer,
				sizeof(optix::WadjetParameters),
				&layer.optix_sbt,
				width, height, 1
			)
		);
		
		CUDA_SYNC_CHECK();

		// Increment number of samples
		layer.launch_params.samples++;

		// Advanced sampling updates
		auto &advanced = layer.launch_params.advanced;

		cuda::copy(advanced.r_temporal_prev, advanced.r_temporal, width * height);
		cuda::copy(advanced.r_spatial_prev, advanced.r_spatial, width * height);
	}

	{
		KOBRA_PROFILE_TASK(Compute postprocessing);

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
			width, height, 0
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

	// KOBRA_PROFILE_PRINT();
}

// Render to the presentable framebuffer
// TODO: custom extent
void render(Wadjet &layer,
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
			*layer.render_pass,
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
		*layer.pipeline
	);

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*layer.ppl, 0, {*layer.dset}, {}
	);

	// Draw and end
	cmd.draw(6, 1, 0, 0);
	cmd.endRenderPass();
}

}

}
