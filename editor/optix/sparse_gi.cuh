#pragma once

// OptiX headers
#include <optix.h>

// Engine headers
#include "include/vertex.hpp"
#include "include/cuda/material.cuh"

#include "../path_tracer.cuh"

// Launch info for the G-buffer raytracer
struct SparseGIParameters {
        // Acceleration structure
        OptixTraversableHandle handle;

        // Global parameters
        float time;
        bool reset;
        bool dirty;

        // Previous camera matrices
        glm::mat4 previous_view;
        glm::mat4 previous_projection;
        float4 *previous_position;

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
        cudaSurfaceObject_t uv_surface;
        cudaSurfaceObject_t index_surface;

        // List of all materials
        kobra::cuda::_material *materials;

        // Lighting information
        struct {
                AreaLight *lights;
                uint count;
                uint triangle_count;
        } area;

        struct {
                cudaTextureObject_t texture;
                bool enabled;
        } sky;
};
