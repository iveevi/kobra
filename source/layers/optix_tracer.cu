// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

// Engine headers
#include "../../include/cuda/error.cuh"
#include "../../include/cuda/alloc.cuh"
#include "../../include/layers/optix_tracer.cuh"
#include "../../include/layers/optix_tracer_common.cuh"
#include "../../include/camera.hpp"
#include "../../include/texture_manager.hpp"
#include "../../include/formats.hpp"
#include "../../include/cuda/color.cuh"

#include <stb_image_write.h>

namespace kobra {

namespace layers {

static void generate_mesh_data
		(const kobra::Raytracer *raytracer,
		const Transform &transform,
		optix_rt::HitGroupData &data)
{
	const Mesh &mesh = raytracer->get_mesh();

	std::vector <float3> vertices(mesh.vertices());
	std::vector <float2> uvs(mesh.vertices());
	std::vector <uint3> triangles(mesh.triangles());
	
	std::vector <float3> normals(mesh.vertices());
	std::vector <float3> tangents(mesh.vertices());
	std::vector <float3> bitangents(mesh.vertices());

	int vertex_index = 0;
	int uv_index = 0;
	int triangle_index = 0;
	
	int normal_index = 0;
	int tangent_index = 0;
	int bitangent_index = 0;

	for (const auto &submesh : mesh.submeshes) {
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

		for (int j = 0; j < submesh.triangles(); j++) {
			triangles[triangle_index++] = {
				submesh.indices[j * 3 + 0],
				submesh.indices[j * 3 + 1],
				submesh.indices[j * 3 + 2]
			};
		}
	}

	data.vertices = cuda::make_buffer(vertices);
	data.texcoords = cuda::make_buffer(uvs);

	data.normals = cuda::make_buffer(normals);
	data.tangents = cuda::make_buffer(tangents);
	data.bitangents = cuda::make_buffer(bitangents);

	data.triangles = cuda::make_buffer(triangles);
}

static void generate_submesh_data
		(const Submesh &submesh,
		const Transform &transform,
		optix_rt::HitGroupData &data)
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

	data.vertices = cuda::make_buffer(vertices);
	data.texcoords = cuda::make_buffer(uvs);

	data.normals = cuda::make_buffer(normals);
	data.tangents = cuda::make_buffer(tangents);
	data.bitangents = cuda::make_buffer(bitangents);

	data.triangles = cuda::make_buffer(triangles);
}

const std::vector <DSLB> OptixTracer::_dslb_render = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

template <class T>
struct Record {
	__align__ (OPTIX_SBT_RECORD_ALIGNMENT)
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	T data;
};

typedef Record <optix_rt::RayGenData>     RayGenSbtRecord;
typedef Record <optix_rt::MissData>       MissSbtRecord;
typedef Record <optix_rt::HitGroupData>   HitGroupSbtRecord;

inline float3 to_f3(const glm::vec3 &v)
{
	return make_float3(v.x, v.y, v.z);
}

__forceinline__
__host__ __device__
uint32_t to_ui32(uchar4 v)
{
	// Reversed
	return (v.w << 24) | (v.z << 16) | (v.y << 8) | v.x;
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::stringstream ss;
	ss << level << std::setw(20) << tag;
	logger(ss.str(), Log::AUTO, "OPTIX") << message << std::endl;
}

static cudaTextureObject_t import_vulkan_texture(const vk::raii::Device &device, const ImageData &img)
{
	// Create a CUDA texture out of the Vulkan image
	cudaExternalMemoryHandleDesc ext_mem_desc {};
	ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	ext_mem_desc.handle.fd = img.get_memory_handle(device);
	ext_mem_desc.size = img.get_size();

	// Import the external memory
	cudaExternalMemory_t tex_mem;
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaImportExternalMemory(&tex_mem, &ext_mem_desc));

	// Create a mipmapped array for the texture
	cudaExternalMemoryMipmappedArrayDesc mip_desc {};
	mip_desc.flags = 0;
	mip_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	mip_desc.numLevels = 1;
	mip_desc.offset = 0;
	mip_desc.extent = make_cudaExtent(
		img.extent.width,
		img.extent.height, 0
	);

	cudaMipmappedArray_t mip_array;
	CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mip_array, tex_mem, &mip_desc));

	// Create the final texture object
	cudaResourceDesc res_desc {};
	res_desc.resType = cudaResourceTypeMipmappedArray;
	res_desc.res.mipmap.mipmap = mip_array;

	cudaTextureDesc tex_desc {};
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	tex_desc.normalizedCoords = true;
	tex_desc.filterMode = cudaFilterModeLinear;

	cudaTextureObject_t tex_obj;
	CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

	return tex_obj;
}

// Set environment map
void OptixTracer::environment_map(const std::string &path)
{
	// First load the environment map
	_v_environment_map = &TextureManager::load_texture(
		*_ctx.phdev,
		*_ctx.device,
		path, true
	);

	// Update miss group record
	MissSbtRecord miss_record;
	miss_record.data.bg_color = float3 {0.0f, 0.0f, 0.0f};
	miss_record.data.bg_tex = import_vulkan_texture(*_ctx.device, *_v_environment_map);

	OPTIX_CHECK(optixSbtRecordPackHeader(_programs.miss_radiance, &miss_record));
	cuda::copy(_optix_miss_sbt, &miss_record, 1);
}

////////////
// Render //
////////////

void OptixTracer::render(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const ECS &ecs, const RenderArea &ra)
{
	// Get camera and camera transform
	Camera camera;
	Transform camera_transform;
	bool found_camera = false;

	std::vector <const kobra::Raytracer *> raytracers;
	std::vector <Transform> raytracer_transforms;
	std::vector <const Transform *> raytracer_transforms_ptr;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	bool dirty_raytracers = false;
	int raytracers_index = 0;

	// Iterate over all entities
	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <Camera> (i)) {
			camera = ecs.get <Camera> (i);
			camera_transform = ecs.get <Transform> (i);
			found_camera = true;
		}

		if (ecs.exists <kobra::Raytracer> (i)) {
			// TODO: account for changing transforms
			const kobra::Raytracer *raytracer = &ecs.get <kobra::Raytracer> (i);

			if (raytracers_index >= _c_raytracers.size())
				dirty_raytracers = true;
			else if (_c_raytracers[raytracers_index] != raytracer)
				dirty_raytracers = true;
			// TODO: also check for content changes in the component
			raytracer_transforms.push_back(ecs.get <Transform> (i));
			raytracer_transforms_ptr.push_back(&ecs.get <Transform> (i));
			raytracers.push_back(raytracer);
			raytracers_index++;
		}

		// TODO: check dirty lights
		if (ecs.exists <Light> (i)) {
			const Light *light = &ecs.get <Light> (i);

			if (light->type == Light::eArea) {
				lights.push_back(&ecs.get <Light> (i));
				light_transforms.push_back(&ecs.get <Transform> (i));
			}
		}
	}

	// Dirty means reset samples
	bool dirty = (_cached.camera_transform != camera_transform);
	if (dirty) {
		_cached.camera_transform = camera_transform;
		_accumulated = 0;
	}

	if (dirty_raytracers) {
		KOBRA_LOG_FILE(Log::INFO) << "Need to rebuild AS\n";
		_c_raytracers = raytracers;
		_c_transforms = raytracer_transforms;
		_cached.lights = lights;
		_cached.light_transforms = light_transforms;

		_cached.submeshes.clear();
		_cached.submesh_transforms.clear();

		for (int i = 0; i < _c_raytracers.size(); i++) {
			const kobra::Raytracer *r = _c_raytracers[i];
			for (const auto &s : r->get_mesh().submeshes) {
				_cached.submeshes.push_back(&s);
				_cached.submesh_transforms.push_back(raytracer_transforms_ptr[i]);
			}
		}

		_optix_build();
	}

	// Launch OptiX with the given camera
	_optix_update_materials();
	_optix_trace(camera, camera_transform);

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

/////////////////////
// Private methods //
/////////////////////

// #define KOPTIX_DEBUG

void OptixTracer::_initialize_optix()
{
	// Storage for logs
	static char log[1024];
	static size_t sizeof_log = sizeof(log);

	// Initialize CUDA
	CUDA_CHECK( cudaFree( 0 ) );

	// Initialize the OptiX API, loading all API entry points
	OPTIX_CHECK( optixInit() );

	// Specify context options
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction       = &context_log_cb;
	options.logCallbackLevel          = 4;

	// Associate a CUDA context (and therefore a specific GPU) with this
	// device context
	CUcontext cuCtx = 0;  // zero means take the current context

	OPTIX_CHECK(optixDeviceContextCreate( cuCtx, &options, &_optix_ctx));

	// Create the OptiX module
	OptixPipelineCompileOptions pipeline_compile_options = {};

	{
		OptixModuleCompileOptions module_compile_options = {};

#ifdef KOPTIX_DEBUG

#pragma message "CUDA debug enabled"

		module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#endif

		pipeline_compile_options.usesMotionBlur        = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		pipeline_compile_options.numPayloadValues      = 3;
		pipeline_compile_options.numAttributeValues    = 3;

#ifdef KOPTIX_DEBUG

#pragma message"OptiX debug enabled"

		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG
			| OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
			| OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

#else

		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

#endif

		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
		pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		size_t      inputSize  = 0;
		std::string input = kobra::common::read_file("./bin/ptx/optix_rt.ptx");
		inputSize = input.size();

		size_t sizeof_log = sizeof( log );

		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
					_optix_ctx,
					&module_compile_options,
					&pipeline_compile_options,
					input.c_str(),
					inputSize,
					log,
					&sizeof_log,
					&_optix_module
					));
	}

	// Create program groups
	{
		// Default program group option
		OptixProgramGroupOptions program_group_options = {};

		// TODO: reate all at once
		OptixProgramGroupDesc raygen_program_desc = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
			.raygen = {
				.module = _optix_module,
				.entryFunctionName = "__raygen__rg"
			}
		};

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			_optix_ctx,
			&raygen_program_desc, 1,
			&program_group_options,
			log, &sizeof_log,
			&_programs.raygen
		));

		// Miss programs
		OptixProgramGroupDesc miss_program_desc = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
			.miss = {
				.module = _optix_module,
			}
		};
	
		// Radiance miss program
		miss_program_desc.miss.entryFunctionName = "__miss__radiance";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			_optix_ctx,
			&miss_program_desc, 1,
			&program_group_options,
			log, &sizeof_log,
			&_programs.miss_radiance
		));

		// Shadow miss program
		miss_program_desc.miss.entryFunctionName = "__miss__shadow";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			_optix_ctx,
			&miss_program_desc, 1,
			&program_group_options,
			log, &sizeof_log,
			&_programs.miss_shadow
		));

		// Hit programs
		OptixProgramGroupDesc hitgroup_program_desc = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {
				.moduleCH = _optix_module,
			}
		};

		// Radiance hit program
		hitgroup_program_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			_optix_ctx,
			&hitgroup_program_desc, 1,
			&program_group_options,
			log, &sizeof_log,
			&_programs.hit_radiance
		));

		// Shadow hit program
		hitgroup_program_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			_optix_ctx,
			&hitgroup_program_desc, 1,
			&program_group_options,
			log, &sizeof_log,
			&_programs.hit_shadow
		));
	}

	//
	// Link pipeline
	//
	_optix_pipeline = nullptr;
	{
		const int max_trace_depth = 15;

		OptixProgramGroup program_groups[] = {
			_programs.raygen,
			_programs.hit_radiance,
			_programs.hit_shadow,
			_programs.miss_radiance,
			_programs.miss_shadow
		};

		OptixPipelineLinkOptions pipeline_link_options = {};
		
		pipeline_link_options.maxTraceDepth          = max_trace_depth;
		pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

		OPTIX_CHECK_LOG( optixPipelineCreate(
					_optix_ctx,
					&pipeline_compile_options,
					&pipeline_link_options,
					program_groups,
					sizeof( program_groups ) / sizeof( program_groups[0] ),
					log,
					&sizeof_log,
					&_optix_pipeline
					) );

		OptixStackSizes stack_sizes = {};
		for( auto& prog_group : program_groups )
		{
			OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
					0,  // maxCCDepth
					0,  // maxDCDEpth
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state, &continuation_stack_size ) );
		OPTIX_CHECK( optixPipelineSetStackSize( _optix_pipeline, direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state, continuation_stack_size,
					2  // maxTraversableDepth
					) );
	}

	/////////////////////////////////
	// Set up shader binding table //
	/////////////////////////////////

	// Ray generation
	CUdeviceptr  raygen_record;
	const size_t raygen_record_size = sizeof( RayGenSbtRecord );
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
	RayGenSbtRecord rg_sbt;
	OPTIX_CHECK( optixSbtRecordPackHeader(_programs.raygen, &rg_sbt ) );
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( raygen_record ),
				&rg_sbt,
				raygen_record_size,
				cudaMemcpyHostToDevice
			      ) );

	// Ray miss records
	std::vector <MissSbtRecord> miss_sbt_records {
		MissSbtRecord {.data = {0.6f, 0.6f, 0.6f}}, {}
	};
		
	OPTIX_CHECK(optixSbtRecordPackHeader(
		_programs.miss_radiance,
		&miss_sbt_records[0]
	));

	OPTIX_CHECK(optixSbtRecordPackHeader(
		_programs.miss_shadow,
		&miss_sbt_records[1]
	));

	_optix_miss_sbt = cuda::make_buffer_ptr(miss_sbt_records);

	// Ray closest hit program
	_optix_hg_sbt = cuda::alloc(sizeof(HitGroupSbtRecord));
	
	HitGroupSbtRecord hg_sbt;

	OPTIX_CHECK(optixSbtRecordPackHeader(_programs.hit_radiance, &hg_sbt));
	cuda::copy(_optix_hg_sbt, &hg_sbt, 1);

	_optix_sbt = OptixShaderBindingTable {};
	_optix_sbt.raygenRecord                = raygen_record;
	_optix_sbt.missRecordBase              = _optix_miss_sbt;
	_optix_sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
	_optix_sbt.missRecordCount             = 2;
	_optix_sbt.hitgroupRecordBase          = _optix_hg_sbt;
	_optix_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
	_optix_sbt.hitgroupRecordCount         = 1;

	// Optix denoiser
	OptixDenoiserOptions denoiser_options = {};
	denoiser_options.guideAlbedo = 1;
	denoiser_options.guideNormal = 1;

	OPTIX_CHECK(optixDenoiserCreate(_optix_ctx,
		OPTIX_DENOISER_MODEL_KIND_AOV,
		&denoiser_options,
		&_optix_denoiser
	));

	// Optix denoiser size
	OptixDenoiserSizes denoiser_sizes;
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(
		_optix_denoiser,
		width, height,
		&denoiser_sizes
	));

	int scratch_size = std::max(
		denoiser_sizes.withOverlapScratchSizeInBytes,
		denoiser_sizes.withoutOverlapScratchSizeInBytes
	);

	_buffers.denoiser_state = std::move(cuda::BufferData(denoiser_sizes.stateSizeInBytes));
	_buffers.denoiser_scratch = std::move(cuda::BufferData(scratch_size));

	// Create stream for OptiX
	CUDA_CHECK(cudaStreamCreate(&_optix_stream));

	// Set up denoiser
	OPTIX_CHECK(optixDenoiserSetup(_optix_denoiser, _optix_stream,
		width, height,
		_buffers.denoiser_state.dev(),
		_buffers.denoiser_state.size(),
		_buffers.denoiser_scratch.dev(),
		_buffers.denoiser_scratch.size()
	));

	KOBRA_LOG_FUNC(Log::OK) << "Initialized OptiX and relevant structures" << std::endl;
}

// TODO: also add an optix_update method
void OptixTracer::_optix_build()
{
	// Use default options for simplicity.  In a real use case we would want to
	// enable compaction, etc
	OptixAccelBuildOptions gas_accel_options = {};
	gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	gas_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	std::vector <OptixTraversableHandle> instance_gas(_cached.submeshes.size());
	std::vector <OptixTraversableHandle> light_gas(_cached.lights.size());
	
	// Flags
	const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

	KOBRA_LOG_FUNC(Log::INFO) << "Building GAS for instances (# = "
		<< _cached.submeshes.size() << ")" << std::endl;

	// TODO: CACHE the vertices for the sbts

	for (int i = 0; i < _cached.submeshes.size(); i++) {
		const Submesh *s = _cached.submeshes[i];

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
			vertices.push_back(to_f3(p));
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
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			_optix_ctx, &gas_accel_options,
			&build_input, 1,
			&gas_buffer_sizes
		));
		
		KOBRA_LOG_FUNC(Log::INFO) << "GAS buffer sizes: " << gas_buffer_sizes.tempSizeInBytes
			<< " " << gas_buffer_sizes.outputSizeInBytes << std::endl;

		d_gas_output = cuda::alloc(gas_buffer_sizes.outputSizeInBytes);
		d_gas_tmp = cuda::alloc(gas_buffer_sizes.tempSizeInBytes);

		OptixTraversableHandle handle;
		OPTIX_CHECK(optixAccelBuild(_optix_ctx,
			0, &gas_accel_options,
			&build_input, 1,
			d_gas_tmp, gas_buffer_sizes.tempSizeInBytes,
			d_gas_output, gas_buffer_sizes.outputSizeInBytes,
			&handle, nullptr, 0
		));

		instance_gas[i] = handle;

		// Free data at the end
		cuda::free(d_vertices);
		cuda::free(d_triangles);

		cuda::free(d_gas_tmp);
	}

	// Lights (a cube for now)
	Mesh box = Mesh::box({0, 0, 0}, {0.5, 0.01, 0.5});

	std::vector <float3> vertices;
	std::vector <uint3> triangles;

	for (auto s : box.submeshes) {
		for (int j = 0; j < s.indices.size(); j += 3) {
			triangles.push_back({
				s.indices[j],
				s.indices[j + 1],
				s.indices[j + 2]
			});
		}

		for (int j = 0; j < s.vertices.size(); j++) {
			auto p = s.vertices[j].position;
			vertices.push_back(to_f3(p));
		}
	}

	CUdeviceptr d_vertices = cuda::make_buffer_ptr(vertices);
	CUdeviceptr d_triangles = cuda::make_buffer_ptr(triangles);

	// Prepare the instance transform
	for (int i = 0; i < _cached.lights.size(); i++) {
		// Create the build input
		OptixBuildInput build_input {};

		build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

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
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			_optix_ctx, &gas_accel_options,
			&build_input, 1,
			&gas_buffer_sizes
		));
		
		KOBRA_LOG_FUNC(Log::INFO) << "Light GAS buffer sizes: " << gas_buffer_sizes.tempSizeInBytes
			<< " " << gas_buffer_sizes.outputSizeInBytes << std::endl;

		d_gas_output = cuda::alloc(gas_buffer_sizes.outputSizeInBytes);
		d_gas_tmp = cuda::alloc(gas_buffer_sizes.tempSizeInBytes);

		OptixTraversableHandle handle;
		OPTIX_CHECK(optixAccelBuild(_optix_ctx,
			0, &gas_accel_options,
			&build_input, 1,
			d_gas_tmp, gas_buffer_sizes.tempSizeInBytes,
			d_gas_output, gas_buffer_sizes.outputSizeInBytes,
			&handle, nullptr, 0
		));

		light_gas[i] = handle;

		// Free data at the end
		cuda::free(d_gas_tmp);
	}

	// Build instances and top level acceleration structure
	std::vector <OptixInstance> instances;

	for (int i = 0; i < _cached.submeshes.size(); i++) {
		glm::mat4 mat = _cached.submesh_transforms[i]->matrix();

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

	for (int i = 0; i < _cached.lights.size(); i++) {
		// Prepare the instance transform
		glm::mat4 mat = _cached.light_transforms[i]->matrix();

		float transform[12] = {
			mat[0][0], mat[1][0], mat[2][0], mat[3][0],
			mat[0][1], mat[1][1], mat[2][1], mat[3][1],
			mat[0][2], mat[1][2], mat[2][2], mat[3][2]
		};

		OptixInstance instance {};
		memcpy(instance.transform, transform, sizeof(float) * 12);

		// Set the instance handle
		instance.traversableHandle = light_gas[i];
		instance.visibilityMask = 0b10;
		instance.sbtOffset = i + _cached.submeshes.size();

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
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			_optix_ctx, &ias_accel_options,
			&ias_build_input, 1,
			&ias_buffer_sizes
		));

		KOBRA_LOG_FUNC(Log::INFO) << "IAS buffer sizes: " << ias_buffer_sizes.tempSizeInBytes << " " << ias_buffer_sizes.outputSizeInBytes << std::endl;

		// Allocate the IAS
		CUdeviceptr d_ias_output = cuda::alloc(ias_buffer_sizes.outputSizeInBytes);
		CUdeviceptr d_ias_tmp = cuda::alloc(ias_buffer_sizes.tempSizeInBytes);

		// Build the IAS
		OPTIX_CHECK(optixAccelBuild(_optix_ctx,
			0, &ias_accel_options,
			&ias_build_input, 1,
			d_ias_tmp, ias_buffer_sizes.tempSizeInBytes,
			d_ias_output, ias_buffer_sizes.outputSizeInBytes,
			&_optix_traversable, nullptr, 0
		));

		cuda::free(d_ias_tmp);
		cuda::free(d_instances);
	}
}

// Update hit group data with materials
// TODO: also update if transforms change
// TODO: refactor to sbts
void OptixTracer::_optix_update_materials()
{
	static std::vector <HitGroupSbtRecord> hg_sbts;
	static std::vector <optix_rt::QuadLight> quad_lights;
	static std::vector <optix_rt::TriangleLight> triangle_lights;

	// Update quad lights
	if (quad_lights.size() != _cached.lights.size()) {
		quad_lights.resize(_cached.lights.size());

		for (int i = 0; i < quad_lights.size(); i++) {
			const Light *light = _cached.lights[i];
			const Transform *transform = _cached.light_transforms[i];
			
			glm::vec3 a {-0.5f, 0, -0.5f};
			glm::vec3 b {0.5f, 0, -0.5f};
			glm::vec3 c {-0.5f, 0, 0.5f};

			a = transform->apply(a);
			b = transform->apply(b);
			c = transform->apply(c);

			quad_lights[i].a = to_f3(a);
			quad_lights[i].ab = to_f3(b - a);
			quad_lights[i].ac = to_f3(c - a);
			quad_lights[i].intensity
				= to_f3(light->power * light->color);
		}

		KOBRA_LOG_FUNC(Log::INFO) << "Number of area lights: " << quad_lights.size() << std::endl;

		_buffers.quad_lights = (CUdeviceptr) cuda::make_buffer(quad_lights);
	}

	// Update triangle lights
	{
		int count = 0;

		std::vector <const Submesh *> emissive_submeshes;
		for (const Submesh *s : _cached.submeshes) {
			if (s->material.type == eEmissive) {
				emissive_submeshes.push_back(s);
				count += s->triangles();
			}
		}

		if (count != triangle_lights.size()) {
			triangle_lights.clear();

			int i = 0;
			for (const Submesh *s : emissive_submeshes) {
				const Transform *transform =
					_cached.submesh_transforms[i];

				for (int j = 0; j < s->triangles(); j++) {
					uint32_t i0 = s->indices[j * 3 + 0];
					uint32_t i1 = s->indices[j * 3 + 1];
					uint32_t i2 = s->indices[j * 3 + 2];

					glm::vec3 a = transform->apply(s->vertices[i0].position);
					glm::vec3 b = transform->apply(s->vertices[i1].position);
					glm::vec3 c = transform->apply(s->vertices[i2].position);

					optix_rt::TriangleLight light;
					light.a = to_f3(a);
					light.ab = to_f3(b - a);
					light.ac = to_f3(c - a);
					light.intensity = to_f3(s->material.emission);

					triangle_lights.push_back(light);
				}
			}

			// Upload to GPU
			_buffers.tri_lights = (CUdeviceptr) cuda::make_buffer(triangle_lights);
		}
	}

	// Update hit records if necessary
	int required_size = 2 * (_cached.submeshes.size() + _cached.lights.size());
	if (hg_sbts.size() != required_size) {
		hg_sbts.clear();

		// Regular raytracers (submeshes)
		for (int i = 0; i < _cached.submeshes.size(); i++) {
			const Submesh *submesh = _cached.submeshes[i];
			Material mat = submesh->material;

			// Material
			cuda::Material material;
			material.diffuse = to_f3(mat.diffuse);
			material.specular = to_f3(mat.specular);
			material.emission = to_f3(mat.emission);
			material.ambient = to_f3(mat.ambient);
			material.shininess = mat.shininess;
			material.roughness = mat.roughness;
			material.refraction = mat.refraction;
			material.type = mat.type;

			HitGroupSbtRecord hg_sbt {};
			hg_sbt.data.material = material;

			generate_submesh_data(*submesh,
					*_cached.submesh_transforms[i], hg_sbt.data);

			// Import textures if necessary
			// TODO: method?
			if (mat.has_albedo()) {
				const ImageData &diffuse = TextureManager::load_texture(
					_ctx.dev(), mat.albedo_texture
				);

				hg_sbt.data.textures.diffuse = import_vulkan_texture(*_ctx.device, diffuse);
				hg_sbt.data.textures.has_diffuse = true;
			}

			if (mat.has_normal()) {
				const ImageData &normal = TextureManager::load_texture(
					_ctx.dev(), mat.normal_texture
				);

				hg_sbt.data.textures.normal = import_vulkan_texture(*_ctx.device, normal);
				hg_sbt.data.textures.has_normal = true;
			}

			if (mat.has_roughness()) {
				const ImageData &roughness = TextureManager::load_texture(
					_ctx.dev(), mat.roughness_texture
				);

				hg_sbt.data.textures.roughness = import_vulkan_texture(*_ctx.device, roughness);
				hg_sbt.data.textures.has_roughness = true;
			}

			// Lights
			hg_sbt.data.quad_lights = (optix_rt::QuadLight *) _buffers.quad_lights;
			hg_sbt.data.n_quad_lights = quad_lights.size();

			hg_sbt.data.tri_lights = (optix_rt::TriangleLight *) _buffers.tri_lights;
			hg_sbt.data.n_tri_lights = triangle_lights.size();

			OPTIX_CHECK(optixSbtRecordPackHeader(_programs.hit_radiance, &hg_sbt));
			hg_sbts.push_back(hg_sbt);
		}

		// Area lights
		for (int i = 0; i < _cached.lights.size(); i++) {
			HitGroupSbtRecord hg_sbt {};
			hg_sbt.data.quad_lights = (optix_rt::QuadLight *) _buffers.quad_lights;
			hg_sbt.data.n_quad_lights = 1;

			hg_sbt.data.tri_lights = (optix_rt::TriangleLight *) _buffers.tri_lights;
			hg_sbt.data.n_tri_lights = 1;

			hg_sbt.data.material.emission = to_f3(_cached.lights[i]->color);
			hg_sbt.data.material.type = Shading::eEmissive;

			OPTIX_CHECK(optixSbtRecordPackHeader(_programs.hit_radiance, &hg_sbt));
			hg_sbts.push_back(hg_sbt);
		}

		// Duplicate the SBTs for the shadow program
		// TODO: delete hit shadow
		int size = hg_sbts.size();
		for (int i = 0; i < size; i++) {
			HitGroupSbtRecord hg_sbt = hg_sbts[i];
			OPTIX_CHECK(optixSbtRecordPackHeader(_programs.hit_shadow, &hg_sbt));
			hg_sbts.push_back(hg_sbt);
		}

		_optix_hg_sbt = (CUdeviceptr) cuda::make_buffer(hg_sbts);

		// Update SBT
		_optix_sbt.hitgroupRecordBase = _optix_hg_sbt;
		_optix_sbt.hitgroupRecordCount = hg_sbts.size();
		_optix_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);

		KOBRA_LOG_FILE(Log::INFO) << "OptiX: SBT updated\n";
	}
}

// Taken from nvidia's book
static void generate_halton_sequence(int N, int b, std::vector <float> &dst)
{
	int n = 0;
	int d = 1;

	for (int i = 0; i < N; i++) {
		int x = d - n;
		if (x == 1) {
			n = 1;
			d *= b;
		} else {
			int y = d/b;
			while (x <= y)
				y /= b;
			n = (b + 1) * y - x;
		}

		dst[i] = (float) n / (float) d;
	}
}

static void generate_pixel_offsets(int N, std::vector <float> &x, std::vector <float> &y)
{
	static std::vector <int> bases {2, 3, 5, 7, 11, 13};

	srand(time(0));
	int r1 = rand() % bases.size();
	int r2 = rand() % bases.size();

	if (r1 == r2)
		r2 = (r2 + 1) % bases.size();

	int b1 = bases[r1];
	int b2 = bases[r2];

	generate_halton_sequence(N, b1, x);
	generate_halton_sequence(N, b2, y);

	for (int i = 0; i < N; i++) {
		x[i] -= 0.5f;
		y[i] -= 0.5f;
	}
}

// Tone maps:
//	0 for sRGB
//	1 for ACES
__global__ void compute_pixel_values(float4 *pixels, uint32_t *target,
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
	target[out_idx] = to_ui32(color);
}

void OptixTracer::_optix_trace(const Camera &camera, const Transform &transform)
{
	// TODO: refresh every x samples
	static bool first = true;

	// TODO: generate offsets in GPU
	if (first) {
		first = false;
		generate_pixel_offsets(width * height, _buffers.h_xoffsets, _buffers.h_yoffsets);
		cuda::copy(_buffers.xoffset, _buffers.h_xoffsets);
		cuda::copy(_buffers.yoffset, _buffers.h_yoffsets);
	}

	// TODO: keep a persistent param object (and only update what's needed)

	// Settings parameters
	optix_rt::Params params;

	params.spp = samples_per_pixel;
	
	params.pbuffer = (float4 *) _buffers.pbuffer;
	params.nbuffer = (float4 *) _buffers.nbuffer;
	params.abuffer = (float4 *) _buffers.abuffer;
	
	params.xoffset = (float *) _buffers.xoffset;
	params.yoffset = (float *) _buffers.yoffset;

	params.image_width  = width;
	params.image_height = height;

	params.accumulated = _accumulated++;
	params.instances = _cached.submeshes.size() + _cached.lights.size();

	params.handle = _optix_traversable;

	auto uvw = kobra::uvw_frame(camera, transform);

	params.cam_eye      = to_f3(transform.position);

	params.cam_u = to_f3(uvw.u);
	params.cam_v = to_f3(uvw.v);
	params.cam_w = to_f3(uvw.w);

	float ms = timer.elapsed_start();

	params.time = sin(ms * 12.3243f) * cos(1 - ms * 0.123f);

	/// Allocate on the GPU
	CUdeviceptr d_param;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**> (&d_param),
				sizeof(optix_rt::Params)));
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_param ),
				&params, sizeof( params ),
				cudaMemcpyHostToDevice
			      ) );

	OPTIX_CHECK(optixLaunch(_optix_pipeline, _optix_stream, d_param,
				sizeof(optix_rt::Params), &_optix_sbt,
				width, height, 1));
	CUDA_SYNC_CHECK();

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );

	CUdeviceptr d_result = 0;

	if (denoiser_enabled) {
		// Denoise
		OptixImage2D color_input;
		color_input.data = _buffers.pbuffer;
		color_input.width = width;
		color_input.height = height;
		color_input.rowStrideInBytes = width * sizeof(float4);
		color_input.pixelStrideInBytes = sizeof(float4);
		color_input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		OptixImage2D normal_input;
		normal_input.data = _buffers.nbuffer;
		normal_input.width = width;
		normal_input.height = height;
		normal_input.rowStrideInBytes = width * sizeof(float4);
		normal_input.pixelStrideInBytes = sizeof(float4);
		normal_input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		OptixImage2D albedo_input;
		albedo_input.data = _buffers.abuffer;
		albedo_input.width = width;
		albedo_input.height = height;
		albedo_input.rowStrideInBytes = width * sizeof(float4);
		albedo_input.pixelStrideInBytes = sizeof(float4);
		albedo_input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		OptixImage2D output;
		output.data = _buffers.fbuffer;
		output.width = width;
		output.height = height;
		output.rowStrideInBytes = width * sizeof(float4);
		output.pixelStrideInBytes = sizeof(float4);
		output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		// Invoke the denoiser
		OptixDenoiserParams denoiser_params = {};

		OptixDenoiserGuideLayer guide_layer;
		guide_layer.normal = normal_input;
		guide_layer.albedo = albedo_input;

		OptixDenoiserLayer layer;
		layer.input = color_input;
		layer.output = output;

		OPTIX_CHECK(optixDenoiserInvoke(_optix_denoiser, _optix_stream,
			&denoiser_params,
			_buffers.denoiser_state.dev(),
			_buffers.denoiser_state.size(),
			&guide_layer,
			&layer, 1,
			0, 0,
			_buffers.denoiser_scratch.dev(),
			_buffers.denoiser_scratch.size()
		));

		d_result = _buffers.fbuffer;
	} else {
		d_result = _buffers.pbuffer;
	}
	
	// Conversion kernel
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
	compute_pixel_values <<<grid, block>>>
		((float4 *) d_result, (uint32_t *) _buffers.truncated,
		 width, height, tonemapping);

	// TODO: multithread
	if (_output.size() != width * height)
		_output.resize(width * height);

	cuda::copy(_output, _buffers.truncated, width * height);
}

}

}
