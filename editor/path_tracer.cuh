#pragma once

// OptiX headers
#include <optix.h>

// Engine headers
#include "include/vertex.hpp"

// Editor headers
#include "optix_io.cuh"

// Launch info for the G-buffer raytracer
struct PathTracerParameters {
        // Acceleration structure
        OptixTraversableHandle handle;

        // Camera parameters
        float3 U;
        float3 V;
        float3 W;
        float3 origin;
        uint2 resolution;

        // Surface to write onto
        float4 *color;

        // G-buffer source surfaces
        cudaSurfaceObject_t position_surface;
        cudaSurfaceObject_t normal_surface;
        cudaSurfaceObject_t index_surface;

        // Environment map
        cudaTextureObject_t environment_map;

        // IO interface
        OptixIO io;
};
