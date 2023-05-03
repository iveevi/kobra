#pragma once

// Engine headers
#include "include/cuda/alloc.cuh"

// Aliases
using namespace kobra;

struct OptixIO {
        char *buffer;
        int *index;
        int size;
};

// Returns a structure with the device size information
inline OptixIO optix_io_create(int size = 1 << 10)
{
        char *buffer = (char *) cuda::alloc(size);
 
        int *index = (int *) cuda::alloc(sizeof(int));
        int start_index = 0;
        // cuda::copy(index, &start_index, sizeof(int));
        cudaMemcpy(index, &start_index, sizeof(int), cudaMemcpyHostToDevice);

        OptixIO io;
        io.buffer = buffer;
        io.index = index;
        io.size = size;

        return io;
}

// Read from structure to host string
inline std::string optix_io_read(OptixIO *io)
{
        std::string s;
        s.resize(io->size);
        cudaMemcpy(&s[0], io->buffer, io->size, cudaMemcpyDeviceToHost);
        return s;
}

// Clear the buffer (host)
inline void optix_io_clear(OptixIO *io)
{
        int start_index = 0;
        cudaMemcpy(io->index, &start_index, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(io->buffer, 0, io->size);
}

// Write char to buffer
__forceinline__ __device__
void optix_io_write_char(OptixIO *io, char c)
{
        int index = atomicAdd(io->index, 1);
        io->buffer[index] = c;

        // If overflow, loop back to the beginning
        if (index == io->size - 1)
                atomicExch(io->index, 0);
}

__forceinline__ __device__
void optix_io_write_str(OptixIO *io, const char *str)
{
        int index = atomicAdd(io->index, 1);
        const char *c = str;
        while (*c != '\0')
                optix_io_write_char(io, *c++);
}

__forceinline__ __device__
void optix_io_write_int(OptixIO *io, int i)
{
        if (i == 0) {
                optix_io_write_char(io, '0');
                return;
        }

        char str[32] { 0 };

        int digits = 0;
        while (i > 0) {
                str[digits++] = '0' + (i % 10);
                i /= 10;
        }

        for (int j = digits - 1; j >= 0; j--)
                optix_io_write_char(io, str[j]);
}

__forceinline__ __device__
void optix_io_write_float(OptixIO *io, float f)
{
        if (f == 0.0f) {
                optix_io_write_str(io, "0.0f");
                return;
        }

        char str[128] { 0 };

        int digits = 0;
        while (f > 0.0f) {
                str[digits++] = '0' + (int) (f * 10.0f);
                f *= 10.0f;
                f -= (int) f;
        }

        for (int j = digits - 1; j >= 0; j--)
                optix_io_write_char(io, str[j]);
}
