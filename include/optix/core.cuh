#ifndef KOBRA_OPTIX_CORE_H_
#define KOBRA_OPTIX_CORE_H_

// Standard headers
#include <iomanip>
#include <sstream>
#include <vector>

// OptiX headers
#include <optix.h>

// Engine headers
#include "../logger.hpp"
#include "../cuda/error.cuh"
#include "../cuda/math.cuh"
#include "../common.hpp"

// Debugging options
// #define KOBRA_OPTIX_DEBUG

#ifdef KOBRA_OPTIX_DEBUG

#define KOBRA_OPTIX_EXCEPTION_FLAGS \
		OPTIX_EXCEPTION_FLAG_DEBUG \
		| OPTIX_EXCEPTION_FLAG_TRACE_DEPTH \
		| OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW

#define KOBRA_OPTIX_DEBUG_LEVEL OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL
#define KOBRA_OPTIX_OPTIMIZATION_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_0

#else

#define KOBRA_OPTIX_EXCEPTION_FLAGS \
		OPTIX_EXCEPTION_FLAG_NONE

#define KOBRA_OPTIX_DEBUG_LEVEL OPTIX_COMPILE_DEBUG_LEVEL_NONE
#define KOBRA_OPTIX_OPTIMIZATION_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_3

#endif

namespace kobra {

namespace optix {

// Generic SBT record structure
template <class T>
struct Record {
	__align__ (OPTIX_SBT_RECORD_ALIGNMENT)
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	T data;
};

template <>
struct Record <void> {
        __align__ (OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// Pack header
inline void pack_header(const OptixProgramGroup &program, void *r)
{
	OPTIX_CHECK(optixSbtRecordPackHeader(program, &r));
}

template <class T>
void pack_header(const OptixProgramGroup &program, Record <T> &r)
{
	OPTIX_CHECK(optixSbtRecordPackHeader(program, &r));
}

// Packing pointers for 32-bit registers
template <class T>
static KCUDA_INLINE KCUDA_HOST_DEVICE
T *unpack_pointer(uint32_t i0, uint32_t i1)
{
	const uint64_t uptr = static_cast <uint64_t> (i0) << 32 | i1;
	T *ptr = reinterpret_cast <T *> (uptr);
	return ptr;
}

template <class T>
static KCUDA_INLINE KCUDA_HOST_DEVICE
void pack_pointer(T * ptr, uint32_t &i0, uint32_t &i1)
{
	const uint64_t uptr = reinterpret_cast <uint64_t> (ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

// TODO: move to source
static void context_logger
		(unsigned int level,
		const char *tag,
		const char *message,
		void *)
{
	std::stringstream ss;
	ss << level << std::setw(20) << tag;
	logger(ss.str(), Log::AUTO, "OPTIX") << message << std::endl;
}

// Create an OptiX context
inline OptixDeviceContext make_context()
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	// Initialize the OptiX API, loading all API entry points
	OPTIX_CHECK(optixInit());

	// Specify context options
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction       = &context_logger;
	options.logCallbackLevel          = 4;
        // options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

	// Associate CUDA context
	CUcontext cuda_context = 0;

	OptixDeviceContext context = 0;
	OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &options, &context));
        OPTIX_CHECK(optixDeviceContextSetCacheEnabled(context, 0));

	return context;
}

// Load an Optix module
inline OptixModule load_optix_module
		(OptixDeviceContext optix_context,
		const std::string &path,
		const OptixPipelineCompileOptions &pipeline_options,
		const OptixModuleCompileOptions &module_options)
{
	char log[2048];
	size_t sizeof_log = sizeof(log);

	std::string file = common::read_file(path);

	OptixModule module;
	OPTIX_CHECK_LOG(
		optixModuleCreate(
			optix_context,
			&module_options, &pipeline_options,
			file.c_str(), file.size(),
			log, &sizeof_log,
			&module
		)
	);

	return module;
}

// Optix program description macros
#define OPTIX_DESC_RAYGEN(_module, name)			\
	OptixProgramGroupDesc {					\
		.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,	\
		.raygen = {					\
			.module = _module,			\
			.entryFunctionName = name		\
		}						\
	}

#define OPTIX_DESC_MISS(_module, name)				\
	OptixProgramGroupDesc {					\
		.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,		\
		.miss = {					\
			.module = _module,			\
			.entryFunctionName = name		\
		}						\
	}

#define OPTIX_DESC_HIT(module, name)				\
	OptixProgramGroupDesc {					\
		.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,	\
		.hitgroup = {					\
			.moduleCH = module,			\
			.entryFunctionNameCH = name		\
		}						\
	}

// Load Optix programs from a module
inline OptixProgramGroup load_program_group
		(const OptixDeviceContext &optix_context,
		const OptixProgramGroupDesc &desc,
		const OptixProgramGroupOptions &options)
{
	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroup group;
	OPTIX_CHECK_LOG(
		optixProgramGroupCreate(
			optix_context,
			&desc, 1,
			&options,
			log, &sizeof_log,
			&group
		)
	);

	return group;
}

inline void load_program_groups
		(const OptixDeviceContext &optix_context,
		const std::vector <OptixProgramGroupDesc> &descs,
		const OptixProgramGroupOptions &options,
		const std::vector <OptixProgramGroup *> &groups)
{
	KOBRA_ASSERT(
		descs.size() == groups.size(),
		"Number of program group descriptions must match number of groups"
	);

	for (int i = 0; i < descs.size(); i++)
		*groups[i] = load_program_group(optix_context, descs[i], options);
}

// Create and configure OptiX pipeline
OptixPipeline link_optix_pipeline
	(const OptixDeviceContext &,
	const std::vector <OptixProgramGroup> &,
	const OptixPipelineCompileOptions &,
	const OptixPipelineLinkOptions &);

}

}

#endif
