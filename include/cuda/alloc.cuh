#ifndef KOBRA_CUDA_ALLOC_H_
#define KOBRA_CUDA_ALLOC_H_

// Standard headers
#include <array>
#include <vector>

// Engine headers
#include "error.cuh"

namespace kobra {

namespace cuda {

// Allocate
template <class T>
inline T *alloc(size_t size)
{
	T *ptr;
	cudaError_t err = cudaMalloc((void **) &ptr, size * sizeof(T));
	CUDA_CHECK(err);
	return ptr;
}

inline CUdeviceptr alloc(size_t size)
{
	CUdeviceptr ptr;
	cudaError_t err = cudaMalloc((void **) &ptr, size);
	CUDA_CHECK(err);
	return ptr;
}

// Copy
template <class T>
inline void copy(T *dst, T *src, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	cudaError_t err = cudaMemcpy(dst, src, size * sizeof(T), kind);
	CUDA_CHECK(err);
}

template <class T>
inline void copy(T *dst, const std::vector <T> &src, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	copy(dst, src.data(), src.size(), kind);
}

template <class T>
inline void copy(CUdeviceptr dst, T *src, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	cudaError_t err = cudaMemcpy((void *) dst, src, size * sizeof(T), kind);
	CUDA_CHECK(err);
}

template <class T>
inline void copy(CUdeviceptr dst, const std::vector <T> &src, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	copy(dst, src.data(), src.size(), kind);
}

template <class T>
inline void copy(T *dst, CUdeviceptr src, size_t size, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
	cudaError_t err = cudaMemcpy(dst, (void *) src, size * sizeof(T), kind);
	CUDA_CHECK(err);
}

template <class T>
inline void copy(std::vector <T> &dst, CUdeviceptr src, size_t size, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
	dst.resize(size);
	copy(dst.data(), src, size, kind);
}

// Free
template <class T>
inline void free(T *ptr)
{
	cudaError_t err = cudaFree(ptr);
	CUDA_CHECK(err);
}

inline void free(CUdeviceptr ptr)
{
	cudaError_t err = cudaFree((void *) ptr);
	CUDA_CHECK(err);
}

}

}

#endif
