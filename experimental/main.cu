#include <array>
#include <iostream>
#include <sstream>

#include <optix.h>
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/Camera.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include "cuda_output_buffer.h"
#include "common.hpp"

#include "../include/common.hpp"

unsigned int width = 1280;
unsigned int height = 720;

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, 2.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}

int main()
{
        char log[2048]; // For error reporting from OptiX creation functions
	std::cout << "Hello World!" << std::endl;

	// Initialize CUDA with a no-op call to the the CUDA runtime API
	cudaFree(0);

	// Initialize the OptiX API, loading all API entry points 
	optixInit();

	// Specify options for this context. We will use the default options.
	OptixDeviceContextOptions options = {};

	// Associate a CUDA context (and therefore a specific GPU) with this
	// device context
	CUcontext cuCtx = 0; // NULL means take the current active context

	OptixDeviceContext context = nullptr;
	optixDeviceContextCreate( cuCtx, &options, &context );
        
	//
        // accel handling
        //
        OptixTraversableHandle gas_handle;
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
            OPTIX_CHECK( optixAccelComputeMemoryUsage(
                        context,
                        &accel_options,
                        &triangle_input,
                        1, // Number of build inputs
                        &gas_buffer_sizes
                        ) );
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
                        context,
                        0,                  // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,                  // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_gas_output_buffer,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        nullptr,            // emitted property list
                        0                   // num emitted properties
                        ) );

            // We can now free the scratch space buffer used during build and the vertex
            // inputs, since they are not needed by our trivial shading method
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
        }

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
            module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t      inputSize  = 0;
            /* const char* input      = sutil::getInputData("wahoo",
			"experimental", "code.cu", inputSize ); */

	    std::string input = kobra::common::read_file("code.ptx");
	    inputSize = input.size();

            size_t sizeof_log = sizeof( log );

            OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        input.c_str(),
                        inputSize,
                        log,
                        &sizeof_log,
                        &module
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
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
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
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = max_trace_depth;
            pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        log,
                        &sizeof_log,
                        &pipeline
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
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
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

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = 1;
        }

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            sutil::Camera cam;
            configureCamera( cam, width, height );

            Params params;
            params.image        = output_buffer.map();
            params.image_width  = width;
            params.image_height = height;
            params.handle       = gas_handle;
            params.cam_eye      = cam.eye();
            cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();

            output_buffer.unmap();
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        }

	std::string outfile = "out.png";

        //
        // Display results
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
	    sutil::saveImage( outfile.c_str(), buffer, false );
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
}
