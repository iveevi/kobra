#ifndef KOBRA_OPTIX_CORE_H_
#define KOBRA_OPTIX_CORE_H_

// Standard headers
#include <sstream>
#include <iomanip>

// OptiX headers
#include <optix.h>

// Engine headers
#include "../logger.hpp"
#include "../cuda/error.cuh"
#include "../cuda/math.cuh"

namespace kobra {

namespace optix {

// Generic SBT record structure
template <class T>
struct Record {
	__align__ (OPTIX_SBT_RECORD_ALIGNMENT)
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	T data;
};

// Pack header
template <class T>
void pack_header(const OptixProgramGroup &program, Record <T> &r)
{
	OPTIX_CHECK(
		optixSbtRecordPackHeader(
			program, &r
		)
	);
}

// Packing pointers for 32-bit registers
template <class T>
static KCUDA_INLINE KCUDA_HOST_DEVICE
T *unpack_point(uint32_t i0, uint32_t i1)
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
OptixDeviceContext make_context()
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	// Initialize the OptiX API, loading all API entry points
	OPTIX_CHECK(optixInit());

	// Specify context options
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction       = &context_logger;
	options.logCallbackLevel          = 4;

	// Associate CUDA context
	CUcontext cuda_context = 0;

	
	OptixDeviceContext context = 0;
	OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &options, &context));

	return context;
}

}

}

#endif
