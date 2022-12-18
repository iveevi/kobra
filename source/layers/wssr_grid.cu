// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>

// Engine headers
#include "../../include/layers/wssr.cuh"
#include "../../include/camera.hpp"
#include "../../include/cuda/alloc.cuh"
#include "../../include/cuda/cast.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/cuda/interop.cuh"
#include "../../include/ecs.hpp"
#include "../../include/optix/core.cuh"
#include "../../include/profiler.hpp"
#include "../../include/texture_manager.hpp"
#include "../../include/transform.hpp"
#include "../../shaders/raster/bindings.h"

// OptiX Source PTX
#define OPTIX_PTX_FILE "bin/ptx/wssr_grid.ptx"

namespace kobra {

namespace layers {

// SBT record types
using RaygenRecord = optix::Record <int>;
using MissRecord = optix::Record <int>;
using HitRecord = optix::Record <optix::Hit>;

// OptiX compilation configurations
static constexpr OptixPipelineCompileOptions ppl_compile_options = {
	.usesMotionBlur = false,
	.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
	.numPayloadValues = 2,
	.numAttributeValues = 0,
	.exceptionFlags = KOBRA_OPTIX_EXCEPTION_FLAGS,
	.pipelineLaunchParamsVariableName = "parameters",
	.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
};
	
static constexpr OptixModuleCompileOptions module_options = {
	.optLevel = KOBRA_OPTIX_OPTIMIZATION_LEVEL,
	.debugLevel = KOBRA_OPTIX_DEBUG_LEVEL,
};
	
static constexpr OptixPipelineLinkOptions ppl_link_options = {
	.maxTraceDepth = 10,
	.debugLevel = KOBRA_OPTIX_DEBUG_LEVEL,
};

// Load OptiX program groups
static void load_optix_program_groups(GridBasedReservoirs &layer)
{
	// Load programs
	OptixProgramGroupOptions program_options = {};

	// Descriptions of all the programs
	std::vector <OptixProgramGroupDesc> program_descs = {
		OPTIX_DESC_RAYGEN(layer.optix_module, "__raygen__eval"),
		OPTIX_DESC_RAYGEN(layer.optix_module, "__raygen__samples"),
		OPTIX_DESC_HIT(layer.optix_module, "__closesthit__eval"),
		OPTIX_DESC_HIT(layer.optix_module, "__closesthit__samples"),
		OPTIX_DESC_HIT(layer.optix_module, "__closesthit__shadow"),
		OPTIX_DESC_MISS(layer.optix_module, "__miss__ms"),
		OPTIX_DESC_MISS(layer.optix_module, "__miss__shadow")
	};

	// Corresponding program groups
	std::vector <OptixProgramGroup *> program_groups = {
		&layer.optix_programs.raygen,
		&layer.optix_programs.sampling_raygen,
		&layer.optix_programs.hit,
		&layer.optix_programs.sampling_hit,
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
void GridBasedReservoirs::initialize_optix()
{
	// Create the context
	optix_context = optix::make_context();

	// Allocate a stream for the layer
	CUDA_CHECK(cudaStreamCreate(&optix_stream));

	// Now load the module
	optix_module = optix::load_optix_module(
		optix_context, OPTIX_PTX_FILE,
		ppl_compile_options, module_options
	);

	// Load programs
	load_optix_program_groups(*this);

	optix_pipeline = optix::link_optix_pipeline(
		optix_context,
		{
			optix_programs.hit,
			optix_programs.miss,
			optix_programs.raygen,
			optix_programs.sampling_hit,
			optix_programs.sampling_raygen,
			optix_programs.shadow_miss
		},
		ppl_compile_options,
		ppl_link_options
	);

	// Sampling stage SBT
	std::vector <RaygenRecord> sampling_rg_records(1);
	std::vector <MissRecord> sampling_ms_records(1);

	optix::pack_header(optix_programs.sampling_raygen, sampling_rg_records[0]);

	optix::pack_header(optix_programs.miss, sampling_ms_records[0]);

	CUdeviceptr d_raygen_sbt = cuda::make_buffer_ptr(sampling_rg_records);
	CUdeviceptr d_miss_sbt = cuda::make_buffer_ptr(sampling_ms_records);

	sampling_sbt.raygenRecord = d_raygen_sbt;
	
	sampling_sbt.missRecordBase = d_miss_sbt;
	sampling_sbt.missRecordCount = sampling_ms_records.size();
	sampling_sbt.missRecordStrideInBytes = sizeof(MissRecord);

	// Evaluation stage SBT
	std::vector <RaygenRecord> eval_rg_records(1);
	std::vector <MissRecord> eval_ms_records(2);
	
	optix::pack_header(optix_programs.raygen, eval_rg_records[0]);

	optix::pack_header(optix_programs.miss, eval_ms_records[0]);
	optix::pack_header(optix_programs.shadow_miss, eval_ms_records[1]);

	d_raygen_sbt = cuda::make_buffer_ptr(eval_rg_records);
	d_miss_sbt = cuda::make_buffer_ptr(eval_ms_records);

	eval_sbt.raygenRecord = d_raygen_sbt;
	
	eval_sbt.missRecordBase = d_miss_sbt;
	eval_sbt.missRecordCount = eval_ms_records.size();
	eval_sbt.missRecordStrideInBytes = sizeof(MissRecord);

	// Configure launch parameters
	int width = extent.width;
	int height = extent.height;

	launch_params.resolution = {
		extent.width,
		extent.height
	};

	launch_params.traversable = 0;
	launch_params.envmap = 0;
	launch_params.has_envmap = false;
	launch_params.samples = 0;

	// Lights (set to null, etc)
	launch_params.lights.quad_count = 0;
	launch_params.lights.triangle_count = 0;

	// Accumulatoin on by default
	launch_params.accumulate = true;

	// Color buffer (output)
	size_t bytes = width * height * sizeof(float4);

	launch_params.color_buffer = (float4 *) cuda::alloc(bytes);
	launch_params.normal_buffer = (float4 *) cuda::alloc(bytes);
	launch_params.albedo_buffer = (float4 *) cuda::alloc(bytes);
	launch_params.position_buffer = (float4 *) cuda::alloc(bytes);

	// Grib-based reservoirs resources
	auto &gb_ris = launch_params.gb_ris;

	gb_ris.resolution = optix::GBR_RESOLUTION;
	gb_ris.new_samples = cuda::alloc <optix::Reservoir> (width * height);

	optix::Reservoir default_reservoir {
		.sample = optix::GRBSample {},
		.count = 0,
		.weight = 0.0f,
	};

	std::vector <optix::Reservoir>
	reservoir_array(optix::TOTAL_RESERVOIRS, default_reservoir);

	KOBRA_LOG_FILE(Log::INFO) << "Total reservoirs: " << optix::TOTAL_RESERVOIRS
		<< ", equating to " << optix::TOTAL_RESERVOIRS * sizeof(optix::Reservoir) / 1024.0f / 1024.0f
		<< " MB\n";

	// gb_ris.light_reservoirs = cuda::make_buffer(reservoir_array);
	gb_ris.light_reservoirs_old = cuda::make_buffer(reservoir_array);

	d_cell_sizes = cuda::alloc <int> (optix::TOTAL_CELLS);
	d_sample_indices = cuda::alloc <int> (
		optix::TOTAL_CELLS * optix::GBR_CELL_LIMIT
	);

	gb_ris.cell_sizes = d_cell_sizes;
	gb_ris.sample_indices = d_sample_indices;
	gb_ris.reproject = false;

	// Allocate the parameters buffer
	launch_params_buffer = cuda::alloc(
		sizeof(optix::GridBasedReservoirsParameters)
	);

	// Start the timer
	timer.start();

	// Initialize generator
	generator = {};
}

// Create the layer
GridBasedReservoirs::GridBasedReservoirs
		(const Context &context,
		const std::shared_ptr <amadeus::System> &system,
		const std::shared_ptr <MeshMemory> &mesh_memory,
		const vk::Extent2D &extent)
		: m_system(system), m_mesh_memory(mesh_memory)
{
	// TODO: move to the member initializer list
	device = context.device;
	phdev = context.phdev;
	descriptor_pool = context.descriptor_pool;
	this->extent = extent;

	// Initialize OptiX
	initialize_optix();
}

// Set the environment map
void GridBasedReservoirs::set_envmap(const std::string &path)
{
	// First load the environment map
	const auto &map = TextureManager::load_texture(
		*phdev, *device, path, true // TODO: use dev()
	);

	launch_params.envmap = cuda::import_vulkan_texture(*device, map);
	launch_params.has_envmap = true;
}

// Update the light buffers if needed
static void update_light_buffers(GridBasedReservoirs &layer,
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
		if (glm::length(submesh->material.emission) > 0) {
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
					}
				);
			}
		}

		layer.launch_params.lights.triangles = cuda::make_buffer(layer.host.tri_lights);
		layer.launch_params.lights.triangle_count = layer.host.tri_lights.size();
	}
}

// Update the SBT data
static void update_sbt_data(GridBasedReservoirs &layer,
		const std::vector <MeshMemory::Cachelet> &cachelets,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	int submesh_count = submeshes.size();

	std::vector <HitRecord> sampling_hit_records;
	std::vector <HitRecord> eval_hit_records;

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

		hit_record.data.model = submesh_transforms[i]->matrix();
		hit_record.data.material = material;
		
		hit_record.data.triangles = cachelets[i].m_cuda_triangles;
		hit_record.data.vertices = cachelets[i].m_cuda_vertices;

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

		// Push back
		optix::pack_header(layer.optix_programs.sampling_hit, hit_record);
		sampling_hit_records.push_back(hit_record);

		optix::pack_header(layer.optix_programs.hit, hit_record);
		eval_hit_records.push_back(hit_record);
	}

	// Update the SBT
	CUdeviceptr d_hit_records = cuda::make_buffer_ptr(sampling_hit_records);
	
	layer.sampling_sbt.hitgroupRecordBase = d_hit_records;
	layer.sampling_sbt.hitgroupRecordCount = sampling_hit_records.size();
	layer.sampling_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);

	d_hit_records = cuda::make_buffer_ptr(eval_hit_records);

	layer.eval_sbt.hitgroupRecordBase = d_hit_records;
	layer.eval_sbt.hitgroupRecordCount = eval_hit_records.size();
	layer.eval_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);

	layer.launch_params.instances = submesh_count;
	
	KOBRA_LOG_FILE(Log::INFO) << "Updated SBT with " << submesh_count
		<< " submeshes, for total of " << sampling_hit_records.size() << " hit records\n";
}

// Preprocess scene data

// TODO: perform this in a separate command buffer than the main one used to
// present, etc (and separate queue)
void GridBasedReservoirs::preprocess_scene
		(const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{
	// Update the raytracing system
	bool updated = m_system->update(ecs);

	// Set viewing position
	launch_params.camera = cuda::to_f3(transform.position);

	if (first_frame) {
		p_camera = launch_params.camera;
		first_frame = false;
	}
	
	auto uvw = kobra::uvw_frame(camera, transform);

	launch_params.cam_u = cuda::to_f3(uvw.u);
	launch_params.cam_v = cuda::to_f3(uvw.v);
	launch_params.cam_w = cuda::to_f3(uvw.w);

	// Get time
	launch_params.time = timer.elapsed_start();

	// Check if we need to update the lights
	// TODO: callbacks for light changes

	// TODO: method get_entities_with_component <T>
	// TODO: method get_components <T>

	// Check if we need to update the SBT

	// Preprocess the entities
	std::vector <const Renderable *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <Renderable> (i)) {
			const auto *rasterizer = &ecs.get <Renderable> (i);
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

	// TODO: cache light information...

	// Update data if necessary 
	if (updated || launch_params.traversable == 0) {
		// Load the list of all submeshes
		std::vector <MeshMemory::Cachelet> cachelets; // TODO: redo this
							      // method...
		std::vector <const Submesh *> submeshes;
		std::vector <const Transform *> submesh_transforms;

		for (int i = 0; i < rasterizers.size(); i++) {
			const Renderable *rasterizer = rasterizers[i];
			const Transform *transform = rasterizer_transforms[i];

			// Cache the renderables
			// TODO: all update functions should go to a separate methods
			m_mesh_memory->cache_cuda(rasterizer);

			for (int j = 0; j < rasterizer->mesh->submeshes.size(); j++) {
				const Submesh *submesh = &rasterizer->mesh->submeshes[j];

				cachelets.push_back(m_mesh_memory->get(rasterizer, j));
				submeshes.push_back(submesh);
				submesh_transforms.push_back(transform);
			}
		}

		// Update the data
		update_light_buffers(*this,
			lights, light_transforms,
			submeshes, submesh_transforms
		);

		launch_params.traversable = m_system->build_tlas(rasterizers, 1);
		update_sbt_data(*this, cachelets, submeshes, submesh_transforms);

		// Reset the number of samples stored
		launch_params.samples = 0;
	}
}

__global__
void binner(float3 center, float3 delta,
		optix::Reservoir *base, int size,
		int *thread_indices,
		int *sample_indices,
		int *cell_sizes)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= size)
		return;
	
	for (int i = idx; i < size; i += stride) {
		int ri = thread_indices[i];
		optix::Reservoir &res = base[ri];

		// Skip entirely if empty
		if (res.count <= 0)
			continue;

		optix::GRBSample &sample = res.sample;
		float3 pos = sample.source;

		float3 diff = (pos - center) + optix::GBR_SIZE;

		int dim = optix::GRID_RESOLUTION;
		int x = (int) (dim * diff.x / (2 * optix::GBR_SIZE));
		int y = (int) (dim * diff.y / (2 * optix::GBR_SIZE));
		int z = (int) (dim * diff.z / (2 * optix::GBR_SIZE));

		if (x < 0 || x >= dim || y < 0 || y >= dim || z < 0 || z >= dim)
			continue;

		int bin = x + y * dim + z * dim * dim;
		
		// Atomically add to the corresponding cell if space is available
		int *address = &cell_sizes[bin];
		int old = *address;
		int assumed;
		
		if (old >= optix::GBR_CELL_LIMIT)
			continue;

		do {
			assumed = old;
			if (assumed >= optix::GBR_CELL_LIMIT)
				break;

			old = atomicCAS(address, assumed, assumed + 1);
		} while (assumed != old);

		if (assumed < optix::GBR_CELL_LIMIT) {
			int index = assumed + bin * optix::GBR_CELL_LIMIT;
			sample_indices[index] = ri;
		}
	}
}

__global__
void cluster_and_merge
		(optix::Reservoir *dst,
		optix::Reservoir *base,
		const int *const sample_indices,
		const int *const cell_sizes,
		float3 h_seed)
{
	float3 seed = h_seed;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= optix::TOTAL_CELLS)
		return;

	for (int i = idx; i < optix::TOTAL_CELLS; i += stride) {
		int dst_base = i * optix::GBR_RESERVOIR_COUNT;
		int sample_base = i * optix::GBR_CELL_LIMIT;
		int sample_count = cell_sizes[i];

		if (sample_count <= 0)
			continue;

		optix::Reservoir reservoirs[optix::GBR_RESERVOIR_COUNT];
		optix::Reservoir *dst_res = &dst[dst_base];
		
		// First fill empty reservoirs
		int sample_index = 0;
		for (int j = 0; j < optix::GBR_RESERVOIR_COUNT; j++) {
			if (dst[dst_base + j].count <= 0) {
				if (sample_index < sample_count) {
					int index = sample_indices[sample_base + sample_index];
					dst[dst_base + j] = base[index];
					sample_index++;
				}
			}
		}

		if (sample_index >= sample_count)
			continue;

		// Then update the rest stochastically
		// TODO: M-capping
		for (int j = sample_index; j < sample_count; j++) {
			int index = sample_indices[sample_base + j];
			optix::Reservoir &res = base[index];

			// Pick a reservoir to update
			int r = cuda::rand_uniform(optix::GBR_RESERVOIR_COUNT, seed);
			optix::Reservoir &dst_res = dst[dst_base + r];

			// Update the reservoir
			float denom = res.count * res.sample.target;
			float W = (denom > 0.0f) ? res.weight / denom : 0.0f;
			float w = res.sample.target * W * res.count;

			int count = dst_res.count;
			reservoir_update(&dst_res, res.sample, w, seed);
			dst_res.count = count + res.count;
		}
	}
}

// Path tracing computation
void GridBasedReservoirs::render
		(const ECS &ecs,
		const Camera &camera,
		const Transform &transform,
		bool accumulate)
{
	preprocess_scene(ecs, camera, transform);

	// Reset the accumulation state if needed
	if (!accumulate)
		launch_params.samples = 0;

	// Copy parameters to the GPU
	cuda::copy(
		launch_params_buffer,
		&launch_params, 1,
		cudaMemcpyHostToDevice
	);
	
	int width = extent.width;
	int height = extent.height;

	// NOTE: the two tracing stages can be done independently?
	OPTIX_CHECK(
		optixLaunch(
			optix_pipeline,
			optix_stream,
			launch_params_buffer,
			sizeof(optix::GridBasedReservoirsParameters),
			&sampling_sbt,
			width, height, 1
		)
	);
	
	CUDA_SYNC_CHECK();

	float3 camera_delta = launch_params.camera - p_camera;
	{
		// Process new samples
		int total = width * height;
		
		int threads = 256;
		int blocks = (total + threads - 1) / threads;

		// TODO: memset to correct values
		cudaMemset(d_sample_indices, 0, sizeof(int) * optix::TOTAL_CELLS * optix::GBR_CELL_LIMIT);
		cudaMemset(d_cell_sizes, 0, sizeof(int) * optix::TOTAL_CELLS);

		auto &gb_ris = launch_params.gb_ris;

		optix::Reservoir *base = gb_ris.new_samples;
		optix::Reservoir *dst = gb_ris.light_reservoirs_old;

		// Create a random permutation of indices to start with
		// TODO: there is a very large overhead here
		std::vector <int> threads_indices(total);
		for (int i = 0; i < total; i++)
			threads_indices[i] = i;

		std::shuffle(threads_indices.begin(), threads_indices.end(), generator);

		int *d_threads_indices = cuda::make_buffer(threads_indices);

		binner <<<blocks, threads>>> (
			launch_params.camera,
			camera_delta,
			base, total,
			d_threads_indices,
			d_sample_indices,
			d_cell_sizes
		);
		
		CUDA_SYNC_CHECK();

		cuda::free(d_threads_indices);

		cluster_and_merge <<<blocks, threads>>> (
			dst, base,
			d_sample_indices,
			d_cell_sizes,
			float3 {
				(float) timer.elapsed_start(),
				(float) launch_params.samples,
				(float) total,
			}
		);
	}

	// Copy the parameters again
	cuda::copy(
		launch_params_buffer,
		&launch_params, 1,
		cudaMemcpyHostToDevice
	);

	// Execute the second stage
	OPTIX_CHECK(
		optixLaunch(
			optix_pipeline,
			optix_stream,
			launch_params_buffer,
			sizeof(optix::GridBasedReservoirsParameters),
			&eval_sbt,
			width, height, 1
		)
	);
	
	CUDA_SYNC_CHECK();

	// Increment number of samples
	launch_params.samples++;

	// Always copy back the camera position
	p_camera = launch_params.camera;
}

}

}
