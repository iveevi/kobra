// OptiX headers
#include <optix_stubs.h>
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>

// Engine headers
#include "../../include/optix/core.cuh"

namespace kobra {

namespace optix {

OptixPipeline link_optix_pipeline
		(const OptixDeviceContext &optix_context,
		 const std::vector <OptixProgramGroup> &program_groups,
		 const OptixPipelineCompileOptions &ppl_compile_options,
		 const OptixPipelineLinkOptions &ppl_link_options)
{
	static char log[2048];
	static size_t sizeof_log = sizeof(log);

	OptixPipeline pipeline;

	// Create the pipeline and configure it
	OPTIX_CHECK_LOG(
		optixPipelineCreate(
			optix_context,
			&ppl_compile_options,
			&ppl_link_options,
			program_groups.data(),
			program_groups.size(),
			log, &sizeof_log,
			&pipeline
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
			pipeline,
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

	return pipeline;
}

}

}
