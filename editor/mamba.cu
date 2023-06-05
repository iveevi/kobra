// Engine headers
#include "include/cuda/alloc.cuh"

// Local headers
#include "editor/common.hpp"
#include "editor/editor_viewport.cuh"
#include "mamba.cuh"

using namespace kobra;

// OptiX compilation options
static constexpr OptixPipelineCompileOptions pipeline_compile_options = {
	.usesMotionBlur = false,
	.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
	.numPayloadValues = 2,
	.numAttributeValues = 0,
	.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
	// .exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG,
	.pipelineLaunchParamsVariableName = "info",
	.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
};

static constexpr OptixModuleCompileOptions module_compile_options = {
	.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
	.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
	// .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
	// .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
};

static constexpr OptixPipelineLinkOptions pipeline_link_options = {
	.maxTraceDepth = 1,
};

// Constructor
Mamba::Mamba(const OptixDeviceContext &context)
{
        static constexpr const char OPTIX_PTX_FILE[] = "bin/ptx/mamba_shader.o";

        // Load module
        module = optix::load_optix_module(context, OPTIX_PTX_FILE, pipeline_compile_options, module_compile_options);

        // Load programs
        OptixProgramGroupOptions program_options = {};

        // Descriptions of all the programs
        std::vector <OptixProgramGroupDesc> program_descs = {
                OPTIX_DESC_RAYGEN (module, "__raygen__direct_primary"),
                OPTIX_DESC_RAYGEN (module, "__raygen__temporal_reuse"),
                OPTIX_DESC_RAYGEN (module, "__raygen__spatial_reuse"),
                OPTIX_DESC_HIT    (module, "__closesthit__"),
                OPTIX_DESC_MISS   (module, "__miss__"),
        };

        // Corresponding program groups
        std::vector <OptixProgramGroup *> program_groups = {
                &raygen_direct_primary,
                &raygen_direct_temporal,
                &raygen_direct_spatial,
                &closest_hit,
                &miss,
        };

        optix::load_program_groups(
                context,
                program_descs,
                program_options,
                program_groups
        );

        direct_primary_ppl = optix::link_optix_pipeline(context, {
                raygen_direct_primary,
                raygen_direct_temporal,
                raygen_direct_spatial,
                closest_hit,
                miss,
        }, pipeline_compile_options, pipeline_link_options);
        
        direct_temporal_ppl = optix::link_optix_pipeline(context, {
                raygen_direct_temporal,
                closest_hit,
                miss,
        }, pipeline_compile_options, pipeline_link_options);

        // Create shader binding table
        direct_initial_sbt = {};
        direct_temporal_ppl = {};

        // Ray generation
        CUdeviceptr dev_direct_initial_raygen_record;

        optix::Record <void> direct_initial_raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_direct_primary, &direct_initial_raygen_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_direct_initial_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_direct_initial_raygen_record, &direct_initial_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        optix::Record <void> direct_temporal_raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_direct_temporal, &direct_temporal_raygen_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_direct_temporal_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_direct_temporal_raygen_record, &direct_temporal_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        optix::Record <void> direct_spatial_raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_direct_spatial, &direct_spatial_raygen_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_direct_spatial_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_direct_spatial_raygen_record, &direct_spatial_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss, &miss_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_miss_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_miss_record, &miss_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        direct_initial_sbt.raygenRecord = dev_direct_initial_raygen_record;
        direct_initial_sbt.missRecordBase = dev_miss_record;
        direct_initial_sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        direct_initial_sbt.missRecordCount = 1;
        direct_initial_sbt.hitgroupRecordBase = 0;
        direct_initial_sbt.hitgroupRecordStrideInBytes = 0;
        direct_initial_sbt.hitgroupRecordCount = 0;

        // std::memcpy(&direct_temporal_sbt, &direct_initial_sbt, sizeof(OptixShaderBindingTable));

        // direct_temporal_sbt.raygenRecord = dev_direct_temporal_raygen_record;
        
        // Setup parameters
        launch_info = {};
        launch_info.io = optix_io_create();
        launch_info.direct.reservoirs = 0;
        launch_info.direct.Le = 0;

        // Allocate device pointers
        // TODO: for probes, crete lazily on first use
        dev_launch_info = cuda::alloc(sizeof(MambaLaunchInfo));
}

// Final gather functions
struct FinalGather {
        float4 *color;
        float3 *direct;

        vk::Extent2D extent;
};

__global__
void final_gather(FinalGather info)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        int x = index % info.extent.width;
        int y = index / info.extent.width;
        if (x >= info.extent.width || y >= info.extent.height)
                return;

        float4 &color = info.color[index];
        color = make_float4(info.direct[index], 1.0f);
}

// Rendering function
void Mamba::render(EditorViewport *ev,
                const RenderInfo &render_info,
                const std::vector <Entity> &entities,
                const MaterialDaemon *md)
{
        const Camera &camera = render_info.camera;
        const Transform &camera_transform = render_info.camera_transform;
        const vk::Extent2D &extent = ev->extent;
       
        // TODO: pass common rtx instead of ev..
        
        // Handle resizing
        if (resize_queue.size() > 0) {
                vk::Extent2D new_extent = resize_queue.back();
                resize_queue = {};
                
                // if (launch_info.indirect.block_offsets != 0)
                //         CUDA_CHECK(cudaFree((void *) launch_info.indirect.block_offsets));

                if (launch_info.direct.reservoirs != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.direct.reservoirs));
                
                if (launch_info.direct.previous != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.direct.previous));

                if (launch_info.direct.Le != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.direct.Le));

                // TODO: reservoirs should be staggered (size + 1 on each side)
                // so that we can average corners for each pixel
                int size = new_extent.width * new_extent.height;
                launch_info.direct.reservoirs = cuda::alloc <Reservoir <LightInfo>> (size);
                launch_info.direct.previous = cuda::alloc <Reservoir <LightInfo>> (size);
                launch_info.direct.Le = cuda::alloc <float3> (size);

                // Generate block offsets
                // uint N2 = launch_info.indirect.N * launch_info.indirect.N;
                // uint2 nblocks;
                // nblocks.x = 1 + (new_extent.width / launch_info.indirect.N);
                // nblocks.y = 1 + (new_extent.height / launch_info.indirect.N);
                //
                // std::vector <uint> block_offsets(nblocks.x * nblocks.y);
                // std::mt19937 rng;
                // std::uniform_int_distribution <uint> dist(0, N2 - 1);
                // for (uint i = 0; i < block_offsets.size(); i++) {
                //         uint offset = dist(rng);
                //         block_offsets[i] = offset;
                // }
                //
                // launch_info.indirect.block_offsets = cuda::make_buffer(block_offsets);
        }

        // Configure launch parameters
        launch_info.time = ev->common_rtx.timer.elapsed_start();
        launch_info.dirty = render_info.camera_transform_dirty;
        launch_info.reset = ev->render_state.sparse_gi_reset
                        | ev->common_rtx.material_reset
                        | ev->common_rtx.transform_reset
                        | manual_reset;

        // uint N = launch_info.indirect.N;
        // launch_info.counter = (launch_info.counter + 1) % (N * N);
                
        if (launch_info.reset)
                manual_reset = false;

        ev->render_state.sparse_gi_reset = false;

        launch_info.samples++;
        if (launch_info.dirty)
                launch_info.samples = 0;

        // Configure camera axis
        auto uvw = uvw_frame(camera, camera_transform);

        launch_info.camera.U = cuda::to_f3(uvw.u);
        launch_info.camera.V = cuda::to_f3(uvw.v);
        launch_info.camera.W = cuda::to_f3(uvw.w);
        launch_info.camera.origin = cuda::to_f3(render_info.camera_transform.position);
        launch_info.camera.resolution = { extent.width, extent.height };

        // Configure textures and other buffers
        launch_info.position = ev->framebuffer_images->cu_position_surface;
        launch_info.normal = ev->framebuffer_images->cu_normal_surface;
        launch_info.uv = ev->framebuffer_images->cu_uv_surface;
        launch_info.index = ev->framebuffer_images->cu_material_index_surface;

        launch_info.materials = (cuda::_material *) ev->common_rtx.dev_materials;

        launch_info.sky.texture = ev->environment_map.texture;
        launch_info.sky.enabled = ev->environment_map.valid;
       
        // Copy parameters and launch
        cuda::copy(dev_launch_info, &launch_info, 1, cudaMemcpyHostToDevice);
       
        // TODO: parallelize by having one stage for direct, one for indirect
        // (and then for spatil reuse in restir we paralleize with indirect
        // filtering...)
        OPTIX_CHECK(
                optixLaunch(direct_primary_ppl, 0,
                        dev_launch_info,
                        sizeof(MambaLaunchInfo),
                        &direct_initial_sbt, extent.width, extent.height, 1
                )
        );

        CUDA_SYNC_CHECK();
       
        std::memcpy(&direct_temporal_sbt, &direct_initial_sbt, sizeof(OptixShaderBindingTable));
        direct_temporal_sbt.raygenRecord = dev_direct_temporal_raygen_record;

        OPTIX_CHECK(
                optixLaunch(direct_primary_ppl, 0,
                        dev_launch_info,
                        sizeof(MambaLaunchInfo),
                        &direct_temporal_sbt, extent.width, extent.height, 1
                )
        );
        
        CUDA_SYNC_CHECK();
        
        std::memcpy(&direct_spatial_sbt, &direct_initial_sbt, sizeof(OptixShaderBindingTable));
        direct_spatial_sbt.raygenRecord = dev_direct_spatial_raygen_record;

        OPTIX_CHECK(
                optixLaunch(direct_primary_ppl, 0,
                        dev_launch_info,
                        sizeof(MambaLaunchInfo),
                        &direct_spatial_sbt, extent.width, extent.height, 1
                )
        );
        
        CUDA_SYNC_CHECK();

        // Final gather
        FinalGather info;
        info.color = ev->common_rtx.dev_color;
        info.direct = launch_info.direct.Le;
        info.extent = extent;

        // TODO: more advanced parallelization
        uint block_size = 256;
        uint blocks = (extent.width * extent.height + 255) / 256;

        final_gather <<< blocks, block_size >>> (info);
        CUDA_SYNC_CHECK();
        
        // Report any IO exchanges
        std::string io = optix_io_read(&launch_info.io);
        std::cout << "Mamba GI output: \"" << io << "\"" << std::endl;
        optix_io_clear(&launch_info.io);
        
        // Update previous camera state
        launch_info.previous_view = camera.view_matrix(camera_transform);
        launch_info.previous_projection = camera.perspective_matrix();
        launch_info.previous_origin = cuda::to_f3(render_info.camera_transform.position);
}
