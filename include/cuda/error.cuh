#ifndef KOBRA_CUDA_ERROR_H_
#define KOBRA_CUDA_ERROR_H_

// Standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "../logger.hpp"

namespace kobra {

namespace cuda {

class Exception : public std::runtime_error
{
	public:
		Exception( const char* msg )
			: std::runtime_error( msg )
		{
		}

		Exception( OptixResult res, const char* msg )
			: std::runtime_error( createMessage( res, msg ).c_str() )
		{
		}

	private:
		std::string createMessage( OptixResult res, const char* msg )
		{
			std::ostringstream out;
			out << optixGetErrorName( res ) << ": " << msg;
			return out.str();
		}
};

inline void optixCheck( OptixResult res, const char* call, const char* file, unsigned int line )
{
	if( res != OPTIX_SUCCESS )
	{
		std::stringstream ss;
		ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
		throw Exception( res, ss.str().c_str() );
	}
}

inline void optixCheckLog( OptixResult  res,
		const char*  log,
		size_t       sizeof_log,
		size_t       sizeof_log_returned,
		const char*  call,
		const char*  file,
		unsigned int line )
{
	if( res != OPTIX_SUCCESS )
	{
		std::stringstream ss;
		ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
			<< log << ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" ) << '\n';
		throw Exception( res, ss.str().c_str() );
	}
}

inline void optixCheckNoThrow( OptixResult res, const char* call, const char* file, unsigned int line )
{
	if( res != OPTIX_SUCCESS )
	{
		std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
		std::terminate();
	}
}

inline void check_error(cudaError_t error, const char *call, const char *file, unsigned int line)
{
	if (error != cudaSuccess) {
		std::string from = "CUDA: " + function_name(call);
		Logger::error_from(from.c_str()) << cudaGetErrorString(error)
			<< " (" << file << ": " << line << ")\n";
		throw std::runtime_error("CUDA error");
	}
}

inline void sync_check( const char* file, unsigned int line )
{
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::string from = "CUDA: cudaDeviceSynchronize";
		Logger::error_from(from.c_str()) << cudaGetErrorString(error)
			<< " (" << file << ": " << line << ")\n";
		throw std::runtime_error("CUDA error");
	}
}

inline void cudaCheckNoThrow( cudaError_t error, const char* call, const char* file, unsigned int line )
{
	if( error != cudaSuccess )
	{
		std::cerr << "CUDA call (" << call << " ) failed with error: '"
			<< cudaGetErrorString( error ) << "' (" << file << ":" << line << ")\n";
		std::terminate();
	}
}

}

}

#define CUDA_CHECK(call) kobra::cuda::check_error(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() kobra::cuda::sync_check(__FILE__, __LINE__)

// TODO: move these two into optix/
#define OPTIX_CHECK( call )                                                    \
    kobra::cuda::optixCheck( call, #call, __FILE__, __LINE__ )

#define OPTIX_CHECK_LOG( call )                                                \
    kobra::cuda::optixCheckLog( call, log, sizeof( log ), sizeof_log, #call, __FILE__, __LINE__ )

#endif
