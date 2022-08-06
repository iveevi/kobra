// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

// Engine headers
#include "../../include/cuda/error.cuh"
// #include "../../include/layers/tmp_buf.cuh"
#include "../../include/layers/optix_tracer.cuh"
#include "../../include/camera.hpp"

namespace kobra {

namespace layers {

const std::vector <DSLB> OptixTracer::_dslb_render = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
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
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
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
		KOBRA_LOG_FILE(notify) << "Need to rebuild AS\n";
		_c_raytracers = raytracers;
		_c_transforms = raytracer_transforms;
		_optix_build();
	}

	// Launch OptiX with the given camera
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
	
void OptixTracer::_initialize_optix()
{
	// Storage for logs
	static char log[1024];
	size_t sizeof_log = sizeof( log );

	/* cudaFree(0);
	optixInit();

	// Options
	OptixDeviceContextOptions options;

	// Context
	CUcontext cu_ctx = nullptr;

	OptixDeviceContext context = nullptr;
	optixDeviceContextCreate(cu_ctx, &options, &context); */
            
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
	
	/*
        // accel handling
        //
        CUdeviceptr            d_gas_output_buffer;
        {
		// Use default options for simplicity.  In a real use case we would want to
		// enable compaction, etc
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

		// Triangle build input: simple list of three vertices
		const std::array <float3, 3> vertices =
		{ {
			  { -0.5f, -0.5f, 0.0f },
			  {  0.5f, -0.5f, 0.0f },
			  {  0.0f,  0.5f, 0.0f }
		  } };

		const size_t vertices_size = sizeof( float3 )*vertices.size();
		CUdeviceptr d_vertices=0;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
		CUDA_CHECK( cudaMemcpy(
					reinterpret_cast<void*>( d_vertices ),
					vertices.data(),
					vertices_size,
					cudaMemcpyHostToDevice
				      ) );

		// Our build input is a simple list of non-indexed triangle vertices
		const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangle_input = {};
		triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
		triangle_input.triangleArray.vertexBuffers = &d_vertices;
		triangle_input.triangleArray.flags         = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			_optix_ctx,
			&accel_options, &triangle_input,
			1, &gas_buffer_sizes
		));

		CUdeviceptr d_temp_buffer_gas;
		CUDA_CHECK( cudaMalloc(
					reinterpret_cast<void**>( &d_temp_buffer_gas ),
					gas_buffer_sizes.tempSizeInBytes
				      ) );
		CUDA_CHECK( cudaMalloc(
					reinterpret_cast<void**>( &d_gas_output_buffer ),
					gas_buffer_sizes.outputSizeInBytes
				      ) );

		OPTIX_CHECK( optixAccelBuild(
					_optix_ctx,
					0,                  // CUDA stream
					&accel_options,
					&triangle_input,
					1,                  // num build inputs
					d_temp_buffer_gas,
					gas_buffer_sizes.tempSizeInBytes,
					d_gas_output_buffer,
					gas_buffer_sizes.outputSizeInBytes,
					&_optix_traversable,
					nullptr,            // emitted property list
					0                   // num emitted properties
					) );

		// We can now free the scratch space buffer used during build and the vertex
		// inputs, since they are not needed by our trivial shading method
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
	} */

	// Create the OptiX module
	OptixPipelineCompileOptions pipeline_compile_options = {};

	{
		OptixModuleCompileOptions module_compile_options = {};

#ifdef KCUDA_DEBUG

		module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#endif

		pipeline_compile_options.usesMotionBlur        = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.numPayloadValues      = 3;
		pipeline_compile_options.numAttributeValues    = 3;

#ifdef KOPTIX_DEBUG

		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

#else

		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

#endif

		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
		pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		size_t      inputSize  = 0;
		std::string input = kobra::common::read_file("code.ptx");
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
	OptixProgramGroup miss_prog_group     = nullptr;
	OptixProgramGroup hitgroup_prog_group = nullptr;
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
					&miss_prog_group
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
					&hitgroup_prog_group
					) );
	}
        
	//
        // Link pipeline
        //
        _optix_pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

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
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //
        // Set up shader binding table
        //
        {
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

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( hitgroup_record ),
                        &hg_sbt,
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                        ) );

	    _optix_sbt = OptixShaderBindingTable {};
            _optix_sbt.raygenRecord                = raygen_record;
            _optix_sbt.missRecordBase              = miss_record;
            _optix_sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            _optix_sbt.missRecordCount             = 1;
            _optix_sbt.hitgroupRecordBase          = hitgroup_record;
            _optix_sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            _optix_sbt.hitgroupRecordCount         = 1;
        }

	CUDA_CHECK(cudaStreamCreate(&_optix_stream));

	KOBRA_LOG_FILE(ok) << "Initialized OptiX and relevant structures" << std::endl;
}

void OptixTracer::_optix_build()
{
        CUdeviceptr            d_gas_output_buffer;

	// Use default options for simplicity.  In a real use case we would want to
	// enable compaction, etc
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	/* Triangle build input: simple list of three vertices
	const std::array <float3, 3> vertices {
		  float3 { -0.5f, -0.5f, 0.0f },
		  float3 {  0.5f, -0.5f, 0.0f },
		  float3 {  0.0f,  0.5f, 0.0f }
	}; */
	
	std::vector <float3> vertices;

	int ti = 0;
	for (const kobra::Raytracer *rt : _c_raytracers) {
		const Mesh &mesh = rt->get_mesh();
		for (auto s : mesh.submeshes) {
			for (auto i : s.indices) {
				auto p = s.vertices[i].position;
				p = _c_transforms[ti].apply(p);
				vertices.push_back(to_f3(p));
			}
		}
			
		ti++;
	}

	const size_t vertices_size = sizeof( float3 )*vertices.size();
	CUdeviceptr d_vertices=0;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_vertices ),
				vertices.data(),
				vertices_size,
				cudaMemcpyHostToDevice
			      ) );

	// Our build input is a simple list of non-indexed triangle vertices
	const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
	OptixBuildInput triangle_input = {};
	triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
	triangle_input.triangleArray.vertexBuffers = &d_vertices;
	triangle_input.triangleArray.flags         = triangle_input_flags;
	triangle_input.triangleArray.numSbtRecords = 1;

	OptixAccelBufferSizes gas_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		_optix_ctx,
		&accel_options, &triangle_input,
		1, &gas_buffer_sizes
	));

	CUdeviceptr d_temp_buffer_gas;
	CUDA_CHECK( cudaMalloc(
				reinterpret_cast<void**>( &d_temp_buffer_gas ),
				gas_buffer_sizes.tempSizeInBytes
			      ) );
	CUDA_CHECK( cudaMalloc(
				reinterpret_cast<void**>( &d_gas_output_buffer ),
				gas_buffer_sizes.outputSizeInBytes
			      ) );

	OPTIX_CHECK( optixAccelBuild(
				_optix_ctx,
				0,                  // CUDA stream
				&accel_options,
				&triangle_input,
				1,                  // num build inputs
				d_temp_buffer_gas,
				gas_buffer_sizes.tempSizeInBytes,
				d_gas_output_buffer,
				gas_buffer_sizes.outputSizeInBytes,
				&_optix_traversable,
				nullptr,            // emitted property list
				0                   // num emitted properties
				) );

	// We can now free the scratch space buffer used during build and the vertex
	// inputs, since they are not needed by our trivial shading method
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
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
