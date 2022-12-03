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
#include "../../include/layers/basilisk.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/texture_manager.hpp"
#include "../../include/transform.hpp"
#include "../../shaders/raster/bindings.h"
#include "../../include/profiler.hpp"

// OptiX Source PTX
#define OPTIX_PTX_FILE "bin/ptx/basilisk_rt.ptx"
#define OPTIX_RESTIR_PTX_FILE "bin/ptx/basilisk_restir.ptx"
#define OPTIX_VOXEL_PTX_FILE "bin/ptx/basilisk_voxel.ptx"

namespace kobra {

namespace layers {

// SBT record types
using RaygenRecord = optix::Record <int>;
using MissRecord = optix::Record <int>;
using HitRecord = optix::Record <optix::Hit>;

// Load OptiX program groups
// #define KOBRA_OPTIX_DEBUG

static void load_optix_program_groups(Basilisk &layer)
{
	// Load programs
	OptixProgramGroupOptions program_options = {};

	// Descriptions of all the programs
	std::vector <OptixProgramGroupDesc> program_descs = {
		OPTIX_DESC_RAYGEN(layer.optix_module, "__raygen__rg"),
		OPTIX_DESC_HIT(layer.optix_module, "__closesthit__ch"),
		OPTIX_DESC_HIT(layer.optix_restir_module, "__closesthit__restir"),
		OPTIX_DESC_HIT(layer.optix_restir_module, "__closesthit__restir_pt"),
		OPTIX_DESC_HIT(layer.optix_voxel_module, "__closesthit__voxel"),
		OPTIX_DESC_HIT(layer.optix_module, "__closesthit__shadow"),
		OPTIX_DESC_MISS(layer.optix_module, "__miss__ms"),
		OPTIX_DESC_MISS(layer.optix_module, "__miss__shadow")
	};

	// Corresponding program groups
	std::vector <OptixProgramGroup *> program_groups = {
		&layer.optix_programs.raygen,
		&layer.optix_programs.hit,
		&layer.optix_programs.hit_restir,
		&layer.optix_programs.hit_restir_pt,
		&layer.optix_programs.hit_voxel,
		&layer.optix_programs.shadow_hit,
		&layer.optix_programs.miss,
		&layer.optix_programs.shadow_miss
	};

	optix::load_program_groups(
		layer.optix_context,
		program_descs,
		program_options,
		program_groups
	);
}

// Setup and load OptiX things
const int VOXEL_RESOLUTION = 100;

static void initialize_optix(Basilisk &layer)
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

#ifdef KOBRA_OPTIX_DEBUG

	ppl_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG
		| OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
		| OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

#else

	ppl_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

#endif

	ppl_compile_options.pipelineLaunchParamsVariableName = "parameters";
	
	ppl_compile_options.usesPrimitiveTypeFlags =
		OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
	
	ppl_compile_options.traversableGraphFlags =
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

	// Module configuration
	OptixModuleCompileOptions module_options = {};


#ifdef KOBRA_OPTIX_DEBUG

	module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

#else

	module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

#endif

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
	
	std::string restir_file = common::read_file(OPTIX_RESTIR_PTX_FILE);
	OPTIX_CHECK_LOG(
		optixModuleCreateFromPTX(
			layer.optix_context,
			&module_options, &ppl_compile_options,
			restir_file.c_str(), restir_file.size(),
			log, &sizeof_log,
			&layer.optix_restir_module
		)
	);
	
	std::string voxel_file = common::read_file(OPTIX_VOXEL_PTX_FILE);
	OPTIX_CHECK_LOG(
		optixModuleCreateFromPTX(
			layer.optix_context,
			&module_options, &ppl_compile_options,
			voxel_file.c_str(), voxel_file.size(),
			log, &sizeof_log,
			&layer.optix_voxel_module
		)
	);

	// Load programs
	load_optix_program_groups(layer);

	// Create the pipeline and configure it
	OptixPipelineLinkOptions ppl_link_options = {};

	ppl_link_options.maxTraceDepth = 10;

#ifdef KOBRA_OPTIX_DEBUG

	ppl_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#else

	ppl_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

#endif
		
	layer.optix_pipeline = optix::link_optix_pipeline(
		layer.optix_context,
		{
			layer.optix_programs.raygen,
			layer.optix_programs.miss,
			layer.optix_programs.shadow_miss,
			layer.optix_programs.hit,
			layer.optix_programs.hit_restir,
			layer.optix_programs.hit_voxel,
		},
		ppl_compile_options,
		ppl_link_options
	);

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

	params.envmap = 0;
	params.has_envmap = false;
	params.samples = 0;

	// Lights (set to null, etc)
	layer.launch_params.lights.quad_count = 0;
	layer.launch_params.lights.triangle_count = 0;

	// Accumulatoin on by default
	layer.launch_params.accumulate = true;

	// Options (set all to false/0)
	optix::BasiliskOptions &options = layer.launch_params.options;
	memset(&options, 0, sizeof(options));

	// Advanced sampling resources - ReSTIR GI
	float radius = std::min(width, height)/10.0f;

	optix::LightReservoir r_lights_def {
		.sample = {},
		.count = 0,
		.weight = 0.0f,
		.mis = 0.0f
	};

	optix::ReSTIR_Reservoir def {
		.sample = {},
		.count = 0,
		.weight = 0.0f,
		.mis = 0.0f
	};

	std::vector <optix::LightReservoir> r_lights(width * height, r_lights_def);
	std::vector <optix::ReSTIR_Reservoir> r_temporal(width * height, def);
	std::vector <optix::ReSTIR_Reservoir> r_spatial(width * height, def);
	std::vector <float> sampling_radii(width * height, radius);

	params.advanced.r_lights = cuda::make_buffer(r_lights);
	params.advanced.r_lights_prev = cuda::make_buffer(r_lights);
	params.advanced.r_spatial = cuda::make_buffer(r_lights);
	params.advanced.r_spatial_prev = cuda::make_buffer(r_lights);

	// Color buffer (output)
	size_t bytes = width * height * sizeof(float4);

	layer.launch_params.color_buffer = (float4 *) cuda::alloc(bytes);
	layer.launch_params.normal_buffer = (float4 *) cuda::alloc(bytes);
	layer.launch_params.albedo_buffer = (float4 *) cuda::alloc(bytes);
	layer.launch_params.position_buffer = (float4 *) cuda::alloc(bytes);

	// Plus others
	layer.launch_params.kd_tree = nullptr;
	layer.launch_params.kd_nodes = 0;

	// Allocate the parameters buffer
	layer.launch_params_buffer = cuda::alloc(
		sizeof(optix::BasiliskParameters)
	);

	// Start the timer
	layer.timer.start();
}

// Create the layer
// TOOD: all custom extent...
Basilisk Basilisk::make(const Context &context, const vk::Extent2D &extent)
{
	// To return
	Basilisk layer;

	// Extract critical Vulkan structures
	layer.device = context.device;
	layer.phdev = context.phdev;
	layer.descriptor_pool = context.descriptor_pool;
	layer.extent = extent;

	// Initialize OptiX
	initialize_optix(layer);

	// Others (experimental)...
	layer.positions = new float4[extent.width * extent.height];

	// Return
	return layer;
}

// Set the environment map
void set_envmap(Basilisk &layer, const std::string &path)
{
	// First load the environment map
	const auto &map = TextureManager::load_texture(
		*layer.phdev, *layer.device, path, true
	);

	layer.launch_params.envmap = cuda::import_vulkan_texture(*layer.device, map);
	layer.launch_params.has_envmap = true;
}

// Update the light buffers if needed
static void update_light_buffers(Basilisk &layer,
		const std::vector <const Light *> &lights,
		const std::vector <const Transform *> &light_transforms,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
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

		KOBRA_LOG_FUNC(Log::INFO) << "Uploaded " << quad_lights.size()
			<< " quad lights to the GPU\n";
	}

	// Count number of emissive submeshes
	int emissive_count = 0;

	std::vector <std::pair <const Submesh *, int>> emissive_submeshes;
	for (int i = 0; i < submeshes.size(); i++) {
		const Submesh *submesh = submeshes[i];
		if (glm::length(submesh->material.emission) > 0
				|| submesh->material.has_emission()) {
			emissive_submeshes.push_back({submesh, i});
			emissive_count += submesh->triangles();
		}
	}

	if (layer.host.tri_lights.size() != emissive_count) {
		for (const auto &pr : emissive_submeshes) {
			const Submesh *submesh = pr.first;
			const Transform *transform = submesh_transforms[pr.second];

			for (int i = 0; i < submesh->triangles(); i++) {
				uint32_t i0 = submesh->indices[i * 3 + 0];
				uint32_t i1 = submesh->indices[i * 3 + 1];
				uint32_t i2 = submesh->indices[i * 3 + 2];

				glm::vec3 a = transform->apply(submesh->vertices[i0].position);
				glm::vec3 b = transform->apply(submesh->vertices[i1].position);
				glm::vec3 c = transform->apply(submesh->vertices[i2].position);

				layer.host.tri_lights.push_back(
					optix::TriangleLight {
						cuda::to_f3(a),
						cuda::to_f3(b - a),
						cuda::to_f3(c - a),
						cuda::to_f3(submesh->material.emission)
						// cuda::to_f3(glm::vec3 {1, 1, 1})
						// TODO: what if material has
						// textured emission?
					}
				);
			}
		}

		layer.launch_params.lights.triangles = cuda::make_buffer(layer.host.tri_lights);
		layer.launch_params.lights.triangle_count = layer.host.tri_lights.size();

		KOBRA_LOG_FUNC(Log::INFO) << "Uploaded " << layer.host.tri_lights.size()
			<< " triangle lights to the GPU\n";
	}
}

// Build or update acceleration structures for the scene
static void update_acceleration_structure(Basilisk &layer,
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
		
		// KOBRA_LOG_FUNC(Log::INFO) << "GAS buffer sizes: " << gas_buffer_sizes.tempSizeInBytes
		//	<< " " << gas_buffer_sizes.outputSizeInBytes << std::endl;

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
		instance.sbtOffset = optix::eCount * i;
		instance.instanceId = i;

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

		// KOBRA_LOG_FUNC(Log::INFO) << "IAS buffer sizes: " << ias_buffer_sizes.tempSizeInBytes << " " << ias_buffer_sizes.outputSizeInBytes << std::endl;

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

// Compute scene bounds
static void update_scene_bounds(Basilisk &layer,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	// Collect all bounding boxes
	std::vector <BoundingBox> bboxes;

	for (int i = 0; i < submeshes.size(); i++) {
		const Submesh *submesh = submeshes[i];
		const Transform *transform = submesh_transforms[i];

		BoundingBox bbox = submesh->bbox(*transform);
		bboxes.push_back(bbox);
	}

	// Now merge them in pairs
	while (bboxes.size() > 1) {
		std::vector <BoundingBox> new_bboxes;

		for (int i = 0; i < bboxes.size(); i += 2) {
			BoundingBox bbox;

			if (i + 1 < bboxes.size())
				bbox = bbox_union(bboxes[i], bboxes[i + 1]);
			else
				bbox = bboxes[i];

			new_bboxes.push_back(bbox);
		}

		bboxes = new_bboxes;
	}

	float3 min = cuda::to_f3(bboxes[0].min);
	float3 max = cuda::to_f3(bboxes[0].max);
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
static void update_sbt_data(Basilisk &layer,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	int submesh_count = submeshes.size();

	std::vector <HitRecord> hit_records;
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

		if (mat.has_specular()) {
			const ImageData &specular = TextureManager::load_texture(
				*layer.phdev, *layer.device, mat.specular_texture
			);

			hit_record.data.textures.specular
				= cuda::import_vulkan_texture(*layer.device, specular);
			hit_record.data.textures.has_specular = true;
		}

		if (mat.has_emission()) {
			const ImageData &emission = TextureManager::load_texture(
				*layer.phdev, *layer.device, mat.emission_texture
			);

			hit_record.data.textures.emission
				= cuda::import_vulkan_texture(*layer.device, emission);
			hit_record.data.textures.has_emission = true;
		}

		if (mat.has_roughness()) {
			const ImageData &roughness = TextureManager::load_texture(
				*layer.phdev, *layer.device, mat.roughness_texture
			);

			hit_record.data.textures.roughness
				= cuda::import_vulkan_texture(*layer.device, roughness);
			hit_record.data.textures.has_roughness = true;
		}

		// Push back
		optix::pack_header(layer.optix_programs.hit, hit_record);
		hit_records.push_back(hit_record);
		
		optix::pack_header(layer.optix_programs.hit_restir, hit_record);
		hit_records.push_back(hit_record);

		optix::pack_header(layer.optix_programs.hit_restir_pt, hit_record);
		hit_records.push_back(hit_record);
	
		optix::pack_header(layer.optix_programs.hit_voxel, hit_record);
		hit_records.push_back(hit_record);
	}

	// Update the SBT
	CUdeviceptr d_hit_records = cuda::make_buffer_ptr(hit_records);
	
	layer.optix_sbt.hitgroupRecordBase = d_hit_records;
	layer.optix_sbt.hitgroupRecordCount = hit_records.size();
	layer.optix_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);

	layer.launch_params.instances = submesh_count;

	KOBRA_LOG_FUNC(Log::INFO) << "Updated SBT with " << submesh_count
		<< " submeshes, for total of " << hit_records.size() << " hit records\n";
}

// Preprocess scene data

// TODO: perform this in a separate command buffer than the main one used to
// present, etc (and separate queue)
static void preprocess_scene(Basilisk &layer,
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
		update_light_buffers(layer,
			lights, light_transforms,
			submeshes, submesh_transforms
		);

		update_acceleration_structure(layer, submeshes, submesh_transforms);
		update_sbt_data(layer, submeshes, submesh_transforms);
		update_scene_bounds(layer, submeshes, submesh_transforms);

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

// CPU side construction algorithm
inline float get(float3 a, int axis)
{
	if (axis == 0) return a.x;
	if (axis == 1) return a.y;
	if (axis == 2) return a.z;
	return 0.0f;
}

int build_kd_tree_recursive
		(std::vector <float3> &points, int depth,
		std::vector <optix::WorldNode> &nodes,
		int &node_idx, int &res_idx, int parent = -1)
{
	if (points.size() == 0)
		return -1;

	// Curent axis
	int axis = depth % 3;

	// Find the median
	std::sort(points.begin(), points.end(),
		[axis](float3 a, float3 b) {
			return get(a, axis) < get(b, axis);
		}
	);

	int median = points.size() / 2;

	// Create the node
	optix::WorldNode node {
		.axis = axis,
		.split = get(points[median], axis),
		.point = points[median],
		.parent = parent,
	};

	int index = node_idx++;

	// Recurse
	if (points.size() > 1) {
		std::vector <float3> left(points.begin(), points.begin() + median);
		std::vector <float3> right(points.begin() + median + 1, points.end());

		node.left = build_kd_tree_recursive(
			left, depth + 1, nodes,
			node_idx, res_idx, index
		);

		node.right = build_kd_tree_recursive(
			right, depth + 1, nodes,
			node_idx, res_idx, index
		);
	} else {
		node.left = -1;
		node.right = -1;
		node.data = res_idx++;
	}

	nodes[index] = node;
	return index;
}

void build_kd_tree(Basilisk &layer, float4 *point_array, int size)
{
	KOBRA_PROFILE_TASK(Build K-d tree);

	// Gather all valid points
	std::vector <float3> points;
	for (int i = 0; i < size; i++) {
		float4 point = point_array[i];
		if (point.w > 0.0f)
			points.push_back(make_float3(point));
	}

	// Build the tree
	std::vector <optix::WorldNode> nodes(points.size());

	int node_idx = 0;
	int res_idx = 0;

	build_kd_tree_recursive(points, 0, nodes, node_idx, res_idx);

	std::cout << "root node split: " << nodes[0].split << std::endl;
	std::cout << "\tleft = " << nodes[0].left << std::endl;
	std::cout << "\tright = " << nodes[0].right << std::endl;
	std::cout << "# of nodes = " << node_idx << std::endl;
	std::cout << "Size of reservoir" << sizeof(optix::LightReservoir) << std::endl;
	std::cout << "Size of node" << sizeof(optix::WorldNode) << std::endl;
	std::cout << "# of bytes to allocate = "
		<< node_idx * sizeof(optix::WorldNode) << std::endl;

	// Allocate corresponding resources
	int leaves = res_idx;
	size = node_idx;
	
	int total_reservoirs = leaves;

#ifdef WSRIS_HASH_RESOLUION

	total_reservoirs *= WSRIS_HASH_RESOLUION;

#endif
	
	std::vector <optix::LightReservoir> reservoir_data(total_reservoirs);

	optix::LightReservoir *d_reservoirs = cuda::make_buffer(reservoir_data);
	optix::LightReservoir *d_reservoirs_prev = cuda::make_buffer(reservoir_data);

	std::vector <int> lock_data(size, 0);
	std::vector <int *> lock_ptrs(size);

	int *d_lock_data = cuda::make_buffer(lock_data);
	for (int i = 0; i < size; i++) {
		lock_ptrs[i] = d_lock_data;
		d_lock_data++;
	}

	layer.launch_params.kd_tree = cuda::make_buffer(nodes);
	layer.launch_params.kd_reservoirs = d_reservoirs;
	layer.launch_params.kd_reservoirs_prev = d_reservoirs_prev;
	layer.launch_params.kd_locks = cuda::make_buffer(lock_ptrs);
	layer.launch_params.kd_nodes = size;
	layer.launch_params.kd_leaves = leaves;
}

// Path tracing computation
void compute(Basilisk &layer,
		const ECS &ecs,
		const Camera &camera,
		const Transform &transform,
		unsigned int mode,
		bool accumulate)
{
	KOBRA_PROFILE_TASK(HyrbidTracer compute path tracing);

	// Preprocess the scene
	{
		KOBRA_PROFILE_TASK(Update data);

		preprocess_scene(layer, ecs, camera, transform);
	}

	// Set rendering mode
	layer.launch_params.mode = mode;

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
		// TODO: alpha transparency...
		OPTIX_CHECK(
			optixLaunch(
				layer.optix_pipeline,
				layer.optix_stream,
				layer.launch_params_buffer,
				sizeof(optix::BasiliskParameters),
				&layer.optix_sbt,
				width, height, 1
			)
		);
		
		CUDA_SYNC_CHECK();

		auto &advanced = layer.launch_params.advanced;
		int reservoir_size = sizeof(optix::LightReservoir) * width * height;

		// TODO: only copy during corresponding modes...
		// TODO: async??
		CUDA_CHECK(
			cudaMemcpy(
				advanced.r_lights_prev,
				advanced.r_lights,
				reservoir_size,
				cudaMemcpyDeviceToDevice
			)
		);

		/* TODO: Nan errors here...
		CUDA_CHECK(
			cudaMemcpy(
				advanced.r_spatial_prev,
				advanced.r_spatial,
				reservoir_size,
				cudaMemcpyDeviceToDevice
			)
		); */

		// TODO: async tasks...
		// TODO: templatize by transfer type cudamemcpytype
		// cuda::copy(layer.positions, layer.launch_params.position_buffer);
		if (mode == optix::eVoxel && !layer.initial_kd_tree) {
			// TODO: update vs rebuild of k-d tree
			cudaMemcpy(
				layer.positions,
				layer.launch_params.position_buffer,
				width * height * sizeof(float4),
				cudaMemcpyDeviceToHost
			);

			auto task = [&]() {
				build_kd_tree(
					layer,
					layer.positions,
					width * height
				);
			};

			// layer.async_task = new core::AsyncTask(task);
			// layer.async_task->wait();
			task();

			layer.initial_kd_tree = true;
		}

		if (mode == optix::eVoxel && layer.initial_kd_tree) {
			auto &params = layer.launch_params;
			reservoir_size = sizeof(optix::LightReservoir) * params.kd_leaves;

			CUDA_CHECK(
				cudaMemcpy(
					params.kd_reservoirs_prev,
					params.kd_reservoirs,
					reservoir_size,
					cudaMemcpyDeviceToDevice
				)
			);
		}

		/* TODO: construct k-d tree for reservoirs...
		dim3 block(32, 32);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y
		);

		kd_construction <<<grid, block>>> (
			layer.launch_params.position_buffer,
			width, height
		); */

		// Increment number of samples
		layer.launch_params.samples++;
	}

	// KOBRA_PROFILE_PRINT();
}

}

}
