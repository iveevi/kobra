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
        uint counter;

        // Previous camera matrices
        glm::mat4 previous_view;
        glm::mat4 previous_projection;
        float3 previous_origin;
        float4 *previous_position;

        // Camera parameters
        CameraAxis camera;

        // G-buffer source surfaces
        // TODO: texure objects
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

        Sky sky;

        // Direct lighting (ReSTIR)
        Reservoir <DirectLightingSample> *direct_lighting;

        // Indirect lighting caches
        struct {
                uint *block_offsets;

                float4 *screen_irradiance; // (R, G, B, # samples)
                float4 *final_irradiance;
                float *irradiance_samples;

                float4 *irradiance_directions;
                float *direction_samples;

                // Screen space irradiance probes
                // (R, G, B, depth) arranged in Bw x Bh grid
                // (each probe in NxN grid layed out with octahedral projection)
                float4 *screen_probes;
                int N;
        } indirect;

        // IO interface
        OptixIO io;
};
