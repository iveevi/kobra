#ifndef KOBRA_CUDA_CORE_H_
#define KOBRA_CUDA_CORE_H_

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define KCUDA_HOST_DEVICE __host__ __device__
#    define KCUDA_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define KCUDA_HOST_DEVICE
#    define KCUDA_INLINE inline
#    define KCUDA_STATIC_INIT( ... ) = __VA_ARGS__
#endif

#endif
