#pragma once

// OptiX headers
#include <optix.h>

// Engine headers
#include "include/vertex.hpp"
#include "include/cuda/material.cuh"

// Editor headers
#include "optix_io.cuh"
#include "gbuffer_rtx_shader.cuh"

// Light structures
struct AreaLight {
        glm::mat4 model;
        Vertex *vertices;
        uint3 *indices;
        uint triangles;
        float3 emission;
};

// Launch info for the G-buffer raytracer
struct PathTracerParameters {
        // Acceleration structure
        OptixTraversableHandle handle;

        // Global parameters
        float time;
        uint depth;

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
        cuda::_material *materials;

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

        // IO interface
        OptixIO io;
};
