#ifndef KOBRA_CUDA_ERROR_H_
#define KOBRA_CUDA_ERROR_H_

#include <optix.h>
#include <optix_stubs.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK( call ) ::sutil::cudaCheck( call, #call, __FILE__, __LINE__ )

#define CUDA_SYNC_CHECK() ::sutil::cudaSyncCheck( __FILE__, __LINE__ )

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW( call )                                             \
    ::sutil::cudaCheckNoThrow( call, #call, __FILE__, __LINE__ )

#define OPTIX_CHECK( call )                                                    \
    ::sutil::optixCheck( call, #call, __FILE__, __LINE__ )

#define OPTIX_CHECK_LOG( call )                                                \
    ::sutil::optixCheckLog( call, log, sizeof( log ), sizeof_log, #call, __FILE__, __LINE__ )

namespace sutil {

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

inline void cudaCheck( cudaError_t error, const char* call, const char* file, unsigned int line )
{
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << cudaGetErrorString( error ) << "' (" << file << ":" << line << ")\n";
        throw Exception( ss.str().c_str() );
    }
}

inline void cudaSyncCheck( const char* file, unsigned int line )
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
           << cudaGetErrorString( error ) << "' (" << file << ":" << line << ")\n";
        throw Exception( ss.str().c_str() );
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

inline void assertCond( bool result, const char* cond, const char* file, unsigned int line )
{
    if( !result )
    {
        std::stringstream ss;
        ss << file << " (" << line << "): " << cond;
        throw Exception( ss.str().c_str() );
    }
}

inline void assertCondMsg( bool               result,
                           const char*        cond,
                           const std::string& msg,
                           const char*        file,
                           unsigned int       line )
{
    if( !result )
    {
        std::stringstream ss;
        ss << msg << ": " << file << " (" << line << "): " << cond;
        throw Exception( ss.str().c_str() );
    }
}

[[noreturn]] inline void assertFailMsg( const std::string& msg, const char* file, unsigned int line )
{
    std::stringstream ss;
    ss << msg << ": " << file << " (" << line << ')';
    throw Exception( ss.str().c_str() );
}

}  // end namespace sutil

#endif
