#include "../../include/asmodeus/backend.cuh"
#include "../../include/cuda/alloc.cuh"
#include "../../include/cuda/cast.cuh"
#include "../../include/cuda/interop.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/texture_manager.hpp"

// TODO: manage
#include "../../include/optix/parameters.cuh"

namespace kobra {

namespace asmodeus {

// SBT record types
using RaygenRecord = optix::Record <int>;
using MissRecord = optix::Record <int>;
using HitRecord = optix::Record <optix::Hit>;

// Pipeline structure and methods
Backend::Pipeline::Pipeline(int _miss, int _hit)
		: expected_miss(_miss), expected_hit(_hit) {
	miss.resize(expected_miss);
	hit.resize(expected_hit);

	std::fill(miss.begin(), miss.end(), nullptr);
	std::fill(hit.begin(), hit.end(), nullptr);
}

// Launch OptiX kernel
void Backend::Pipeline::launch(CUdeviceptr parameters,
		size_t parameters_size,
		int width, int height)
{
	// TODO: depth?
	OPTIX_CHECK(
		optixLaunch(
			pipeline, stream,
			parameters, parameters_size,
			&sbt, width, height, 1
		)
	);
	
	CUDA_SYNC_CHECK();
}

// Set the program groups
void set_programs(Backend::Pipeline &pipeline,
		OptixDeviceContext context,
		OptixProgramGroup _ray_generation,
		const std::vector <OptixProgramGroup> &_miss,
		const std::vector <OptixProgramGroup> &_hit,
		const OptixPipelineCompileOptions &ppl_compile_options,
		const OptixPipelineLinkOptions &ppl_link_options)
{
	KOBRA_ASSERT(
		_miss.size() == pipeline.expected_miss &&
		_hit.size() == pipeline.expected_hit,
		"Invalid number of program groups"
	);

	// First transfer the program groups
	pipeline.ray_generation = _ray_generation;
	std::copy(_miss.begin(), _miss.end(), pipeline.miss.begin());
	std::copy(_hit.begin(), _hit.end(), pipeline.hit.begin());

	// Now link the program groups into the pipeline
	std::vector <OptixProgramGroup> program_groups {
		pipeline.ray_generation
	};

	for (auto miss : pipeline.miss)
		program_groups.push_back(miss);

	for (auto hit : pipeline.hit)
		program_groups.push_back(hit);

	// Create the pipeline
	pipeline.pipeline = optix::link_optix_pipeline(
		context,
		program_groups,
		ppl_compile_options,
		ppl_link_options
	);
}

// Initialize corresponding SBT records
void initialize_sbt(Backend::Pipeline &pipeline,
		CUdeviceptr raygen_record,
		CUdeviceptr miss_record, size_t miss_record_size,
		const Backend::HitSbtAllocator &hit_sbt_allocator)
{
	// Raygen and miss records
	pipeline.sbt.raygenRecord = raygen_record;

	pipeline.sbt.missRecordBase = miss_record;
	pipeline.sbt.missRecordStrideInBytes = miss_record_size;
	pipeline.sbt.missRecordCount = pipeline.expected_miss;

	// Initialize hit record allocator
	pipeline.hit_sbt_allocator = hit_sbt_allocator;
}

// Backend construction
Backend Backend::make(const Context &context, const BackendType &rtx_backend)
{
	Backend backend;

	// Initialize basic variables
	KOBRA_ASSERT(
		rtx_backend == BackendType::eOptiX,
		"Only OptiX backend is supported for now"
	);

	backend.rtx_backend = rtx_backend;
	backend.device = context.dev();

	// Initialize OptiX
	backend.optix_context = optix::make_context();
	// backend.optix.shaders = shaders;

	return backend;
}

// Create geometry acceleration structure for a submesh
static void construct_gas_instance(const OptixDeviceContext &optix_context,
		const Submesh *submesh, Backend::Instance &instance)
{
	// TODO: make the following static
	// Build acceleration structures
	OptixAccelBuildOptions gas_accel_options = {};
	gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	gas_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
	
	// Flags
	const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

	// Prepare submesh vertices and triangles
	std::vector <float3> vertices;
	std::vector <uint3> triangles;
	
	// TODO: method to generate accel handle from cuda buffers
	// TODO: import data from rasterizer/renderer
	for (int j = 0; j < submesh->indices.size(); j += 3) {
		triangles.push_back({
			submesh->indices[j],
			submesh->indices[j + 1],
			submesh->indices[j + 2]
		});
	}

	for (int j = 0; j < submesh->vertices.size(); j++) {
		auto p = submesh->vertices[j].position;
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
			optix_context,
			&gas_accel_options,
			&build_input, 1,
			&gas_buffer_sizes
		)
	);
	
	d_gas_output = cuda::alloc(gas_buffer_sizes.outputSizeInBytes);
	d_gas_tmp = cuda::alloc(gas_buffer_sizes.tempSizeInBytes);

	OptixTraversableHandle handle;
	OPTIX_CHECK(
		optixAccelBuild(
			optix_context,
			0, &gas_accel_options,
			&build_input, 1,
			d_gas_tmp, gas_buffer_sizes.tempSizeInBytes,
			d_gas_output, gas_buffer_sizes.outputSizeInBytes,
			&handle, nullptr, 0
		)
	);

	instance.handle = handle;

	// Free data at the end
	cuda::free(d_vertices);
	cuda::free(d_triangles);
	cuda::free(d_gas_tmp);
}

// Create the acceleration structure
// TODO: or update it
static void construct_acceleration_structure(Backend &backend)
{
	// Localize backend members
	OptixDeviceContext &optix_context = backend.optix_context;
	std::vector <Backend::Instance> &instances = backend.instances;

	int instance_count = instances.size();

	KOBRA_LOG_FUNC(Log::INFO) << "Building GAS for instances (# = "
		<< instance_count << ")" << std::endl;

	// TODO: CACHE the vertices for the sbts
	for (int i = 0; i < instance_count; i++) {
		construct_gas_instance(optix_context,
			instances[i].submesh, instances[i]
		);
	}
}

// Construct or update the TLAS for a specific pipeline
static void construct_tlas(Backend &backend, Backend::Pipeline &pipeline)
{
	// Localize backend members
	OptixDeviceContext &optix_context = backend.optix_context;
	std::vector <Backend::Instance> &instances = backend.instances;

	// Build instances and top level acceleration structure
	std::vector <OptixInstance> optix_instances;

	int instance_count = instances.size();
	for (int i = 0; i < instance_count; i++) {
		glm::mat4 mat = instances[i].transform->matrix();

		float transform[12] = {
			mat[0][0], mat[1][0], mat[2][0], mat[3][0],
			mat[0][1], mat[1][1], mat[2][1], mat[3][1],
			mat[0][2], mat[1][2], mat[2][2], mat[3][2]
		};

		OptixInstance instance {};
		memcpy(instance.transform, transform, sizeof(float) * 12);

		// Set the instance handle
		instance.traversableHandle = instances[i].handle;
		instance.visibilityMask = 0b1;
		instance.sbtOffset = pipeline.expected_hit * i;
		instance.instanceId = i;

		optix_instances.push_back(instance);
	}

	// Create top level acceleration structure
	CUdeviceptr d_instances = cuda::make_buffer_ptr(instances);

	// TLAS for objects and lights
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
			optix_context, &ias_accel_options,
			&ias_build_input, 1,
			&ias_buffer_sizes
		)
	);

	// Allocate the IAS
	CUdeviceptr d_ias_output = cuda::alloc(ias_buffer_sizes.outputSizeInBytes);
	CUdeviceptr d_ias_tmp = cuda::alloc(ias_buffer_sizes.tempSizeInBytes);

	// Build the IAS
	OPTIX_CHECK(
		optixAccelBuild(optix_context,
			0, &ias_accel_options,
			&ias_build_input, 1,
			d_ias_tmp, ias_buffer_sizes.tempSizeInBytes,
			d_ias_output, ias_buffer_sizes.outputSizeInBytes,
			&pipeline.tlas, nullptr, 0
		)
	);

	// cuda::free(d_ias_tmp);
	// cuda::free(d_instances);
}

// Construct TLAS for all pipelines
static void update_all_tlas(Backend &backend)
{
	for (auto &pipeline : backend.pipelines) {
		construct_tlas(backend, pipeline);
		pipeline.last_update = clock();
	}
}

static void generate_submesh_data
		(const Submesh &submesh,
		const Transform &transform,
		optix::Hit &data)
{
	// TODO: get vertices and indices (and others actually) from rasterizers
	std::vector <float3> vertices(submesh.vertices.size());
	std::vector <float2> uvs(submesh.vertices.size());
	std::vector <uint3> triangles(submesh.triangles());
	
	std::vector <float3> normals(submesh.vertices.size());
	std::vector <float3> tangents(submesh.vertices.size());
	std::vector <float3> bitangents(submesh.vertices.size());
	
	std::cout << "Generating submesh data" << std::endl;
	std::cout << "Vertices: " << vertices.size() << std::endl;
	std::cout << "Triangles: " << triangles.size() << std::endl;
	std::cout << "Normals: " << normals.size() << std::endl;
	std::cout << "Tangents: " << tangents.size() << std::endl;
	std::cout << "Bitangents: " << bitangents.size() << std::endl;

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

// TODO: pass Pipeline instead? create hit sbts in order
static void update_sbt_data(const Backend &backend,
		const Backend::Pipeline &pipeline,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms,
		CUdeviceptr &hit_data, size_t &stride)
{
	std::cout << "[UPDATE SBT DATA] Loaded programs" << std::endl;
	std::cout << "Hit: " << pipeline.hit[0] << std::endl;
	std::cout << "Shadow hit: " << pipeline.hit[1] << std::endl;
	std::cout << "Miss: " << pipeline.miss[0] << std::endl;
	std::cout << "Shadow miss: " << pipeline.miss[1] << std::endl;

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

		// generate_submesh_data(*submesh, *instances[i].transform, hit_record.data);

		// Import textures if necessary
		// TODO: method
		if (mat.has_albedo()) {
			hit_record.data.textures.diffuse
				= import_texture(backend, mat.albedo_texture);
			hit_record.data.textures.has_diffuse = true;
		}

		if (mat.has_normal()) {
			hit_record.data.textures.normal
				= import_texture(backend, mat.normal_texture);
			hit_record.data.textures.has_normal = true;
		}

		if (mat.has_roughness()) {
			hit_record.data.textures.roughness
				= import_texture(backend, mat.roughness_texture);
			hit_record.data.textures.has_roughness = true;
		}
	
		// Push back
		std::cout << "Pushing back hit record; header: "
			<< OPTIX_SBT_RECORD_ALIGNMENT << std::endl;

		// optix::pack_header(pipeline.hit[0], &hit_record);

		optix::pack_header(pipeline.hit[0], hit_record);
		hit_records.push_back(hit_record);
		
		optix::pack_header(pipeline.hit[1], hit_record);
		hit_records.push_back(hit_record);
	}

	// Copy to device
	// hit_data = cuda::make_buffer_ptr(hit_records);
	stride = sizeof(HitRecord);
}

// Construct or update the SBT for a specific pipeline
static void construct_sbt(const Backend &backend,
		Backend::Pipeline &pipeline,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	CUdeviceptr d_sbt = 0;
	size_t stride = 0;

	// update_sbt_data(backend, pipeline, backend.instances, d_sbt, stride);
	update_sbt_data(backend, pipeline,
		submeshes, submesh_transforms,
		d_sbt, stride
	);

	/* pipeline.hit_sbt_allocator(
		backend, pipeline,
		backend.instances, d_sbt, stride
	); */

	KOBRA_ASSERT(stride > 0, "SBT stride is 0");
	pipeline.sbt.hitgroupRecordBase = d_sbt;
	pipeline.sbt.hitgroupRecordStrideInBytes = stride;
	pipeline.sbt.hitgroupRecordCount = backend.instances.size();
}

// Update SBTs for all pipelines
static void update_all_sbt(Backend &backend,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	for (auto &pipeline : backend.pipelines)
		construct_sbt(backend, pipeline, submeshes, submesh_transforms);
}

// Update Asmodeus backend with new submeshes
static void update_from_submeshes(Backend &backend,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform*> &submesh_transforms)
{
	// Assurances
	KOBRA_ASSERT(
		submeshes.size() == submesh_transforms.size(),
		"Submeshes and transforms must be the same size"
	);

	// Prepare instances from each submesh
	int size = submeshes.size();
	for (int i = 0; i < size; i++) {
		Backend::Instance instance {
			.submesh = submeshes[i],
			.transform = submesh_transforms[i],
			.handle = 0,
		};

		backend.instances.push_back(instance);
	}

	// Construct acceleration structure
	// TODO: do we need options for this (i.e. callback?)
	construct_acceleration_structure(backend);
}

// Update the backend with scene data
bool update(Backend &backend, const ECS &ecs)
{
	// Preprocess the entities
	std::vector <const Rasterizer *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	{
		for (int i = 0; i < ecs.size(); i++) {
			// TODO: one unifying renderer component,
			// with options for raytracing, etc
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

	// Check if the objects are dirty
	bool dirty = false;
	for (int i = 0; i < rasterizers.size(); i++) {
		if (i >= backend.scene.c_rasterizers.size()) {
			dirty = true;
			break;
		}

		if (rasterizers[i] != backend.scene.c_rasterizers[i]) {
			dirty = true;
			break;
		}
	}

	// Update the rasterizers
	// TODO: track frame numberand skip this update
	// step over if the frame number is the same as last time
	// TODO: actually better is to track modification state
	// of the ecs
	// ECS ->inverstors.alert() => backend.ecs_alert() ->needs_update = true
	if (dirty) {
		backend.scene.c_rasterizers = rasterizers;

		// Collect all submeshes
		std::vector <const Submesh *> submeshes;
		std::vector <const Transform *> submesh_transforms;
		for (int i = 0; i < rasterizers.size(); i++) {
			const auto *rasterizer = rasterizers[i];
			const auto *transform = rasterizer_transforms[i];

			const Mesh *mesh = rasterizer->mesh;
			for (int j = 0; j < mesh->submeshes.size(); j++) {
				const Submesh *submesh = &mesh->submeshes[j];
				submeshes.push_back(submesh);
				submesh_transforms.push_back(transform);
			}
		}

		// Update the backend
		update_from_submeshes(backend, submeshes, submesh_transforms);
		// update_all_tlas(backend);
		// update_all_sbt(backend);
		// update_all_sbt(backend, submeshes, submesh_transforms);
	}

	// TODO: lazy update the tlas for pipelines,
	// pass ID to pipeline being used (assigned in pipeline constructor)

	return dirty;
}

// Import a Vulkan texture into CUDA
cudaTextureObject_t import_texture(const Backend &backend, const std::string &path)
{
	const ImageData &image = TextureManager::load_texture(backend.device, path);
	return cuda::import_vulkan_texture(*backend.device.device, image);
}

}

}
