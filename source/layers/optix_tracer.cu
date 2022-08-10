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

#include <stb_image_write.h>

namespace kobra {

namespace layers {

const std::vector <DSLB> OptixTracer::_dslb_render = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord <RayGenData>     RayGenSbtRecord;
typedef SbtRecord <MissData>       MissSbtRecord;
typedef SbtRecord <HitGroupData>   HitGroupSbtRecord;

inline float3 to_f3(const glm::vec3 &v)
{
	return make_float3(v.x, v.y, v.z);
}

inline uint32_t to_ui32(uchar4 v)
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

__global__ void check_texture(cudaTextureObject_t tex, size_t width, size_t height)
{

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float u = x / (float) width;
			float v = y / (float) height;
			float4 flt = tex2D <float4> (tex, u, v);
			uint4 color = tex2D <uint4> (tex, u, v);
			printf("%d %d -> %d %d %d %d or %.2f %.2f %.2f %.2f\n",
				x, y, color.x, color.y, color.z, color.w,
				flt.x, flt.y, flt.z, flt.w);
		}
	}
}
	
static int get_memory_handle(const vk::raii::Device &device, const VkDeviceMemory &memory) {
	// TODO: need to ensure that the the VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION is enabled
	int fd = -1;
	VkMemoryGetFdInfoKHR fdInfo {};
	fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	fdInfo.memory = memory;
	fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

	// TODO: load in backend.cpp
	auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(*device, "vkGetMemoryFdKHR");

	std::cout << "Func = " << func << std::endl;
	if (!func) {
		KOBRA_LOG_FUNC(Log::ERROR) << "vkGetMemoryFdKHR not found\n";
		return -1;
	}

	VkResult result = func(*device, &fdInfo, &fd);
	if (result != VK_SUCCESS) {
		KOBRA_LOG_FUNC(Log::ERROR) << "vkGetMemoryFdKHR failed\n";
		return -1;
	}

	return fd;
}

// Set environment map
void OptixTracer::environment_map(const std::string &path)
{
	_v_environment_map = &TextureManager::load_texture(
		*_ctx.phdev,
		*_ctx.device,
		path, true
	);

	// Create a CUDA texture out of the environment map
	cudaExternalMemoryHandleDesc env_tex_desc {};
	env_tex_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	env_tex_desc.handle.fd = get_memory_handle(*_ctx.device,
			*_v_environment_map->memory);
	env_tex_desc.size = _v_environment_map->get_size();

	std::cout << "Size = " << env_tex_desc.size << ", fd = " << env_tex_desc.handle.fd << std::endl;
	std::cout << "\twidth = " << _v_environment_map->extent.width
		<< ", height = " << _v_environment_map->extent.height << std::endl;
	std::cout << "\tsize = " << _v_environment_map->extent.width * _v_environment_map->extent.height * 4 << std::endl;
	
	// cudaExternalMemory_t env_tex_mem;
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaImportExternalMemory(&_dext_env_map, &env_tex_desc));

	cudaExternalMemoryBufferDesc env_tex_buf_desc {};
	env_tex_buf_desc.flags = 0;
	env_tex_buf_desc.offset = 0;
	env_tex_buf_desc.size = env_tex_desc.size;

	CUdeviceptr env_tex_buf;
	CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(
		(void **) &env_tex_buf, _dext_env_map,
		&env_tex_buf_desc
	));

	/* Get data
	uint32_t *env_tex_data = new uint32_t[env_tex_desc.size / sizeof(uint32_t)];
	CUDA_CHECK(cudaMemcpy(env_tex_data, (void *) env_tex_buf, env_tex_desc.size, cudaMemcpyDeviceToHost));

	uint32_t r = 0, g = 0, b = 0, a = 0;
	for (int i = 0; i < 20; i++) {
		uint32_t v = env_tex_data[i];
		r = (v >> 24) & 0xff;
		g = (v >> 16) & 0xff;
		b = (v >> 8) & 0xff;
		a = v & 0xff;
		std::cout << "(" << r << ", " << g << ", " << b << ", " << a << ")\n";
	} */

	/* Save as png
	stbi_flip_vertically_on_write(true);
	stbi_write_png("cuda_env.png",
		_v_environment_map->extent.width,
		_v_environment_map->extent.height,
		4, (void *) env_tex_data,
		_v_environment_map->extent.width * 4
	); */

	// Create bindless texture
	cudaResourceDesc res_desc {};
	res_desc.resType = cudaResourceTypePitch2D;
	res_desc.res.pitch2D.devPtr = (void *) env_tex_buf;
	res_desc.res.pitch2D.desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	res_desc.res.pitch2D.width = _v_environment_map->extent.width;
	res_desc.res.pitch2D.height = _v_environment_map->extent.height;
	res_desc.res.pitch2D.pitchInBytes = 4 * _v_environment_map->extent.width;

	cudaTextureDesc tex_desc {};
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	tex_desc.normalizedCoords = true;
	tex_desc.filterMode = cudaFilterModeLinear;

	/* tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp; */

	// tex_desc.maxAnisotropy = 1;

	CUDA_CHECK(cudaCreateTextureObject(
		&_tex_env_map, &res_desc, &tex_desc, nullptr
	));

	printf("Created texture, now going to check it...\n");
	// check_texture <<<1, 1>>> (_tex_env_map, 20, 20);
	printf("Checked texture\n");
	std::cout << std::endl;

	cudaResourceViewDesc res_view_desc {};
	CUDA_CHECK(cudaGetTextureObjectResourceViewDesc(&res_view_desc,
				_tex_env_map));

	std::cout << "Desc = " << res_view_desc.format << std::endl;
	std::cout << "\twidth = " << res_view_desc.width << std::endl;
	std::cout << "\theight = " << res_view_desc.height << std::endl;

	// Update miss group record
	MissSbtRecord miss_record;
	miss_record.data.bg_color = float3 {0.0f, 0.0f, 0.0f};
	miss_record.data.bg_tex = _tex_env_map;
	miss_record.data.bg_tex_width = _v_environment_map->extent.width;
	miss_record.data.bg_tex_height = _v_environment_map->extent.height;

	OPTIX_CHECK(optixSbtRecordPackHeader(_miss_prog_group, &miss_record));
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
			raytracers.push_back(raytracer);
			raytracers_index++;
		}
	}

	if (dirty_raytracers) {
		KOBRA_LOG_FILE(Log::INFO) << "Need to rebuild AS\n";
		_c_raytracers = raytracers;
		_c_transforms = raytracer_transforms;
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

#define KCUDA_DEBUG
#define KOPTIX_DEBUG

void OptixTracer::_initialize_optix()
{
	// Storage for logs
	static char log[1024];
	size_t sizeof_log = sizeof( log );

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
	OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &_optix_ctx) );

	// Create the OptiX module
	OptixPipelineCompileOptions pipeline_compile_options = {};

	{
		OptixModuleCompileOptions module_compile_options = {};

#ifdef KCUDA_DEBUG

#warning "CUDA debug enabled"

		module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#endif

		pipeline_compile_options.usesMotionBlur        = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		pipeline_compile_options.numPayloadValues      = 3;
		pipeline_compile_options.numAttributeValues    = 3;

#ifdef KOPTIX_DEBUG

#warning "OptiX debug enabled"

		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

#else

		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

#endif

		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
		pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		size_t      inputSize  = 0;
		std::string input = kobra::common::read_file("./bin/ptx/optix_rt.ptx");
		inputSize = input.size();

		size_t sizeof_log = sizeof( log );

		OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
					_optix_ctx,
					&module_compile_options,
					&pipeline_compile_options,
					input.c_str(),
					inputSize,
					log,
					&sizeof_log,
					&_optix_module
					) );
	}

	//
	// Create program groups
	//
	OptixProgramGroup raygen_prog_group   = nullptr;
	
	{
		OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

		OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
		raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module            = _optix_module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		OPTIX_CHECK_LOG( optixProgramGroupCreate(
					_optix_ctx,
					&raygen_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&raygen_prog_group
					) );

		OptixProgramGroupDesc miss_prog_group_desc  = {};
		miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module            = _optix_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		sizeof_log = sizeof( log );
		OPTIX_CHECK_LOG( optixProgramGroupCreate(
					_optix_ctx,
					&miss_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&_miss_prog_group
					) );

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH            = _optix_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		sizeof_log = sizeof( log );
		OPTIX_CHECK_LOG( optixProgramGroupCreate(
					_optix_ctx,
					&hitgroup_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&_hitgroup_prog_group
					) );
	}

	//
	// Link pipeline
	//
	_optix_pipeline = nullptr;
	{
		const uint32_t    max_trace_depth  = 1;
		OptixProgramGroup program_groups[] = { raygen_prog_group,
			_miss_prog_group, _hitgroup_prog_group };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth          = max_trace_depth;
		pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		size_t sizeof_log = sizeof( log );
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
	OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( raygen_record ),
				&rg_sbt,
				raygen_record_size,
				cudaMemcpyHostToDevice
			      ) );

	// Ray miss program
	_optix_miss_sbt = cuda::alloc(sizeof(MissSbtRecord));
	
	MissSbtRecord ms_sbt;
	ms_sbt.data = {0.6f, 0.6f, 0.9f};

	OPTIX_CHECK(optixSbtRecordPackHeader(_miss_prog_group, &ms_sbt));
	cuda::copy(_optix_miss_sbt, &ms_sbt, 1);

	// Ray closest hit program
	_optix_hg_sbt = cuda::alloc(sizeof(HitGroupSbtRecord));
	
	HitGroupSbtRecord hg_sbt;
	hg_sbt.data.material_count = 2;

	OPTIX_CHECK(optixSbtRecordPackHeader(_hitgroup_prog_group, &hg_sbt));
	cuda::copy(_optix_hg_sbt, &hg_sbt, 1);

	_optix_sbt = OptixShaderBindingTable {};
	_optix_sbt.raygenRecord                = raygen_record;
	_optix_sbt.missRecordBase              = _optix_miss_sbt;
	_optix_sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
	_optix_sbt.missRecordCount             = 1;
	_optix_sbt.hitgroupRecordBase          = _optix_hg_sbt;
	_optix_sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
	_optix_sbt.hitgroupRecordCount         = 1;

	// Create stream
	CUDA_CHECK(cudaStreamCreate(&_optix_stream));

	KOBRA_LOG_FUNC(Log::OK) << "Initialized OptiX and relevant structures" << std::endl;
}

// TODO: also add an optix_update method
void OptixTracer::_optix_build()
{
	// Use default options for simplicity.  In a real use case we would want to
	// enable compaction, etc
	OptixAccelBuildOptions gas_accel_options = {};
	gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	gas_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	std::vector <OptixTraversableHandle> instance_gas(_c_raytracers.size());
	
	// Flags
	const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

	for (int i = 0; i < _c_raytracers.size(); i++) {
		// Prepare the instance transform
		glm::mat4 mat = _c_transforms[i].matrix();

		float transform[12] = {
			mat[0][0], mat[1][0], mat[2][0], mat[3][0],
			mat[0][1], mat[1][1], mat[2][1], mat[3][1],
			mat[0][2], mat[1][2], mat[2][2], mat[3][2]
		};

		// Prepare instance vertices and triangles
		std::vector <float3> vertices;
		std::vector <uint3> triangles;
		
		// TODO: raytracer method
		const Mesh &mesh = _c_raytracers[i]->get_mesh();

		for (auto s : mesh.submeshes) {
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

		// Create the build input
		OptixBuildInput build_input {};

		build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		CUdeviceptr d_vertices = cuda::alloc(vertices.size() * sizeof(float3));
		CUdeviceptr d_triangles = cuda::alloc(triangles.size() * sizeof(uint3));

		// TODO: alloc and memcpy in one step function
		CUdeviceptr d_transform = cuda::alloc(12 * sizeof(float));

		cuda::copy(d_vertices, vertices);
		cuda::copy(d_triangles, triangles);
		cuda::copy(d_transform, transform, 12);

		OptixBuildInputTriangleArray &triangle_array = build_input.triangleArray;
		triangle_array.vertexFormat	= OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_array.numVertices	= vertices.size();
		triangle_array.vertexBuffers	= &d_vertices;

		triangle_array.indexFormat	= OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangle_array.numIndexTriplets	= triangles.size();
		triangle_array.indexBuffer	= d_triangles;

		triangle_array.flags		= triangle_input_flags;
		triangle_array.numSbtRecords	= 1;

		// Set the transform
		// triangle_array.transformFormat	= OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
		// triangle_array.preTransform	= d_transform;

		// Build GAS
		CUdeviceptr d_gas_output;
		CUdeviceptr d_gas_tmp;

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			_optix_ctx, &gas_accel_options,
			&build_input, 1,
			&gas_buffer_sizes
		));

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
		cuda::free(d_gas_tmp);
		// cuda::free(d_vertices);
		// cuda::free(d_triangles);
		// cuda::free(d_transform); // TODO: can be freed after all instances are built
	}

	// Build instances and top level acceleration structure
	std::vector <OptixInstance> instances(_c_raytracers.size());

	for (int i = 0; i < _c_raytracers.size(); i++) {
		// Prepare the instance transform
		// TODO: keep in a separate array
		glm::mat4 mat = _c_transforms[i].matrix();

		float transform[12] = {
			mat[0][0], mat[1][0], mat[2][0], mat[3][0],
			mat[0][1], mat[1][1], mat[2][1], mat[3][1],
			mat[0][2], mat[1][2], mat[2][2], mat[3][2]
		};

		memcpy(instances[i].transform, transform, sizeof(float) * 12);

		// Set the instance handle
		instances[i].sbtOffset = 0;
		instances[i].traversableHandle = instance_gas[i];
		instances[i].visibilityMask = 0xFF;
	}

	// Copy the instances to the device
	CUdeviceptr d_instances = cuda::alloc(instances.size() * sizeof(OptixInstance));
	cuda::copy(d_instances, instances);

	// Create the top level acceleration structure
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

	// Free the memory
	// cuda::free(d_instances);
	// cuda::free(d_ias_tmp);
}

// Update hit group data with materials
void OptixTracer::_optix_update_materials()
{
	// TODO: avoid doing this every frame
	size_t n_materials = _c_raytracers.size();

	std::vector <HitGroupData::Material> materials(n_materials);
	for (int i = 0; i < _c_raytracers.size(); i++) {
		 Material mat = _c_raytracers[i]->get_material();
		 materials[i].diffuse = to_f3(mat.diffuse);
		 materials[i].specular = to_f3(mat.specular);
		 materials[i].emission = to_f3(mat.emission);
	}

	// Turn into CUDA device memory
	// TODO: be more conservative with the memory
	if (n_materials * sizeof(HitGroupData::Material) > _d_materials_size) {
		if (_d_materials != 0)
			cuda::free(_d_materials);
		
		_d_materials = cuda::alloc(sizeof(HitGroupData::Material) * n_materials);
	}

	cuda::copy(_d_materials, materials);

	// Update hit group data
	HitGroupSbtRecord hg_sbt;
	hg_sbt.data.material_count = n_materials;
	hg_sbt.data.materials = (HitGroupData::Material *) _d_materials;

	OPTIX_CHECK(optixSbtRecordPackHeader(_hitgroup_prog_group, &hg_sbt));
	cuda::copy(_optix_hg_sbt, &hg_sbt, 1);
}

void OptixTracer::_optix_trace(const Camera &camera, const Transform &transform)
{
	Params params;
	params.image        = _result_buffer.dev <uchar4> ();
	params.image_width  = width;
	params.image_height = height;
	params.handle       = _optix_traversable;
	params.cam_eye      = to_f3(transform.position);

	auto uvw = kobra::uvw_frame(camera, transform);
	params.cam_u = to_f3(uvw.u);
	params.cam_v = to_f3(uvw.v);
	params.cam_w = to_f3(uvw.w);

	CUdeviceptr d_param;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_param ),
				&params, sizeof( params ),
				cudaMemcpyHostToDevice
			      ) );

	OPTIX_CHECK( optixLaunch( _optix_pipeline, _optix_stream, d_param,
				sizeof( Params ), &_optix_sbt, width, height,
				1) );
	CUDA_SYNC_CHECK();

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );

	// Copy result to buffer
	std::vector <uchar4> ptr = _result_buffer.download <uchar4> ();
	// uchar4 *ptr = _output_buffer.getHostPointer();

	_output.resize(width * height);
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int inv_y = height - y - 1;
			_output[x + inv_y * width] = to_ui32(ptr[x + y * width]);
		}
	}
}

}

}
