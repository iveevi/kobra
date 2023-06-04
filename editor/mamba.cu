#include "mamba.cuh"
#include "include/cuda/alloc.cuh"

using namespace kobra;

// OptiX compilation options
static constexpr OptixPipelineCompileOptions pipeline_compile_options = {
	.usesMotionBlur = false,
	.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
	.numPayloadValues = 2,
	.numAttributeValues = 0,
	.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
	.pipelineLaunchParamsVariableName = "info",
	.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
};

static constexpr OptixModuleCompileOptions module_compile_options = {
	.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
	.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
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
                OPTIX_DESC_RAYGEN (module, "__raygen__"),
                OPTIX_DESC_HIT    (module, "__closesthit__"),
                OPTIX_DESC_MISS   (module, "__miss__"),
        };

        // Corresponding program groups
        std::vector <OptixProgramGroup *> program_groups = {
                &ray_generation,
                &closest_hit,
                &miss,
        };

        optix::load_program_groups(
                context,
                program_descs,
                program_options,
                program_groups
        );

        pipeline = optix::link_optix_pipeline(context, {
                ray_generation,
                closest_hit,
                miss,
        }, pipeline_compile_options, pipeline_link_options);

        // Create shader binding table
        sbt = {};

        // Ray generation
        CUdeviceptr dev_raygen_record;

        optix::Record <void> raygen_record;
        optix::pack_header(ray_generation, &raygen_record);

        dev_raygen_record = (CUdeviceptr) cuda::alloc <optix::Record <void>> (1);
        cuda::copy(dev_raygen_record, &raygen_record, 1, cudaMemcpyHostToDevice);

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        optix::pack_header(miss, &miss_record);

        dev_miss_record = (CUdeviceptr) cuda::alloc <optix::Record <void>> (1);
        cuda::copy(dev_miss_record, &miss_record, 1, cudaMemcpyHostToDevice);

        sbt.raygenRecord = dev_raygen_record;
        sbt.missRecordBase = dev_miss_record;
        sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = 0;
        sbt.hitgroupRecordStrideInBytes = 0;
        sbt.hitgroupRecordCount = 0;
}
