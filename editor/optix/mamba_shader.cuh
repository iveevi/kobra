#pragma once

// Standard headers
#include <random>

// OptiX headers
#include <optix.h>

// Engine headers
#include "common.cuh"
#include "include/cuda/material.cuh"
#include "include/cuda/random.cuh"
#include "include/vertex.hpp"

// Sampled lighting information
struct LightInfo {
        float3 position;
        float3 normal;
        float3 emission;
        float area;
        bool sky;
};

// Irradiance probe
struct IrradianceProbe {
        constexpr static int size = 4;

        // Layed out using octahedral projection
        float3 values[size * size];
        float pdfs[size * size];
        float depth[size * size];
        float3 normal;
};

// Launch info for the G-buffer raytracer
struct MambaLaunchInfo {
        // Acceleration structure
        OptixTraversableHandle handle;

        // Global parameters
        bool dirty;
        bool reset;
        float time;
        int samples;
        uint counter;

        // Previous camera matrices
        glm::mat4 previous_view;
        glm::mat4 previous_projection;
        float3 previous_origin;

        // Camera parameters
        CameraAxis camera;

        // G-buffer information
        cudaSurfaceObject_t position;
        cudaSurfaceObject_t normal;
        cudaSurfaceObject_t uv;
        cudaSurfaceObject_t index;

        // List of all materials
        kobra::cuda::_material *materials;

        // Lighting information
        struct {
                AreaLight *lights;
                uint count;
                uint triangle_count;
        } area;

        Sky sky;

        // Direct lighting (ReSTIR)
        struct {
                Reservoir <LightInfo> *reservoirs;
                Reservoir <LightInfo> *previous;
                float3 *Le;
        } direct;

        // Indirect lighting caches
        struct {
                uint *block_offsets;
                float3 *Le;
                float3 *wo;
        } indirect;

        // IO interface
        OptixIO io;
};
