#ifndef KOBRA_CUDA_BUFFER_DATA_H_
#define KOBRA_CUDA_BUFFER_DATA_H_

// Standard headers
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// Engine headers
#include "error.cuh"
#include "../logger.hpp"

namespace kobra {

namespace cuda {

class BufferData {
	size_t			_size = 0;
	char			*_device_ptr = nullptr;
public:
	// Default constructor
	BufferData() = default;

	// Constructor
	BufferData(size_t size) {
		resize(size);
	}

	// Destructor
	~BufferData() {
		if (_device_ptr != nullptr)
			CUDA_CHECK(cudaFree(_device_ptr));
	}

	// No copy
	BufferData(const BufferData &) = delete;
	BufferData &operator=(const BufferData &) = delete;

	// Move
	BufferData(BufferData &&other) {
		_size = other._size;
		_device_ptr = other._device_ptr;
		other._size = 0;
		other._device_ptr = nullptr;
	}

	BufferData &operator=(BufferData &&other) {
		_size = other._size;
		_device_ptr = other._device_ptr;
		other._size = 0;
		other._device_ptr = nullptr;
		return *this;
	}

	// Get size
	size_t size() const {
		return _size;
	}

	// Get device pointer
	template <class T = void>
	T *dev() {
		return reinterpret_cast <T *> (_device_ptr);
	}

	CUdeviceptr dev() {
		return reinterpret_cast <CUdeviceptr> (_device_ptr);
	}

	// Upload data to device
	template <class T>
	bool upload(const std::vector <T> &data, size_t offset = 0, bool auto_resize = true) {
		static constexpr char size_msg[] = "Buffer size is smaller than data size";

		// Resize status
		bool resized = false;

		// Assertions
		KOBRA_ASSERT(data.size() * sizeof(T) <= _size || auto_resize,
			std::string(size_msg) + " (data size = " + std::to_string(data.size() * sizeof(T))
			+ ", buffer size = " + std::to_string(_size) + ")"
		);

		if (data.size() * sizeof(T) > _size) {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

			KOBRA_LOG_FUNC(Log::ERROR) << size_msg << " (size = " << _size
				<< ", data size = " << data.size() * sizeof(T)
				<< ")" << std::endl;

#endif

			// Resize buffer
			resize(data.size() * sizeof(T));
			resized = true;
		}

		// Upload data
		// TODO: async versions?
		CUDA_CHECK(cudaMemcpy(_device_ptr + offset,
			data.data(), data.size() * sizeof(T),
			cudaMemcpyHostToDevice
		));

		return resized;
	}

	template <class T>
	bool upload(const T *const data, size_t size_, size_t offset = 0, bool auto_resize = true) {
		static constexpr char size_msg[] = "Buffer size is smaller than data size";

		// Resize status
		bool resized = false;

		// Assertions
		KOBRA_ASSERT(size_ <= _size || auto_resize,
			std::string(size_msg) + " (data size = " + std::to_string(size_)
			+ ", buffer size = " + std::to_string(_size) + ")"
		);

		if (size_ > _size) {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

			KOBRA_LOG_FUNC(Log::ERROR) << size_msg << " (size = " << size_
				<< ", buffer size = " << _size << ")" << std::endl;

#endif

			// Resize buffer
			resize(size_);
			resized = true;
		}

		// Upload data
		CUDACHECK(cudaMemcpy(_device_ptr + offset,
			data, size_,
			cudaMemcpyHostToDevice
		));

		return resized;
	}

	// Get buffer data
	template <class T>
	void download(std::vector <T> &v) {
		v.resize(_size/sizeof(T));
		CUDA_CHECK(cudaMemcpy(v.data(), _device_ptr, _size, cudaMemcpyDeviceToHost));
	}

	template <class T>
	std::vector <T> download() const {
		std::vector <T> v;
		v.resize(_size/sizeof(T));
		CUDA_CHECK(cudaMemcpy(v.data(), _device_ptr, _size, cudaMemcpyDeviceToHost));
		return v;
	}

	void resize(size_t size) {
		// SKip unnecessary resizing
		if (_size == size)
			return;

		// Reset device pointer
		if (_device_ptr) {
			cudaFree(_device_ptr);
			_device_ptr = nullptr;
		}

		// Realloc device pointer
		if (size > 0) {
			CUDA_CHECK(cudaMalloc(&_device_ptr, size));
			_size = size;
		} else {
			KOBRA_LOG_FUNC(Log::ERROR) << "Invalid size: " << size << std::endl;
			_size = 0;
		}
	}
};

}

}

#endif
