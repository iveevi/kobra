#pragma once

// OptiX headers
#include <optix.h>

// Engine headers
#include "include/vertex.hpp"
#include "include/cuda/material.cuh"

// Editor headers
#include "optix_io.cuh"

// Launch info for the G-buffer raytracer
struct GBufferParameters {
        // Acceleration structure
        OptixTraversableHandle handle;

        // Camera parameters
        float3 U;
        float3 V;
        float3 W;
        float3 origin;
        uint2 resolution;

        // Surfaces to write onto
        cudaSurfaceObject_t position_surface;
        cudaSurfaceObject_t normal_surface;
        cudaSurfaceObject_t uv_surface;
        cudaSurfaceObject_t index_surface;

        // List of materials
        cuda::_material *materials;

        // IO interface
        OptixIO io;
};

struct Hit {
        kobra::Vertex *vertices;
        glm::mat4 model;
        uint3 *triangles;
        uint32_t index;
};
