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

// Launch info for the G-buffer raytracer
struct SparseGIParameters {
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

        // G-buffer textures
        struct {
                cudaTextureObject_t position;
                cudaTextureObject_t normal;
                cudaTextureObject_t uv;
                cudaTextureObject_t index;
        } gbuffer;

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
        float3 *direct_lighting;

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
