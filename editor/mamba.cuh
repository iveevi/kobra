#pragma once

// Engine headers
#include "include/optix/core.cuh"

// Local headers
#include "editor/optix/mamba_shader.cuh"
#include "include/daemons/material.hpp"
#include "include/system.hpp"

// Forward declarations
struct EditorViewport;
struct RenderInfo;

// Mamba global illumination
struct Mamba {
        // CUDA resources
        CUstream stream_direct = 0;
        CUstream stream_indirect = 0;

        // OptiX resources
        OptixModule module = 0;

        OptixPipeline direct_ppl = 0;

        OptixProgramGroup raygen_direct_primary = 0;
        OptixProgramGroup raygen_direct_temporal = 0;
        OptixProgramGroup raygen_direct_spatial = 0;

        OptixProgramGroup closest_hit = 0;
        OptixProgramGroup miss = 0;

        OptixShaderBindingTable direct_initial_sbt = {};
        OptixShaderBindingTable direct_temporal_sbt = {};
        OptixShaderBindingTable direct_spatial_sbt = {};

        // Kernel parameters
        MambaLaunchInfo launch_info = {};

        // Device pointers
        // TODO: irradiance probes...
        CUdeviceptr dev_launch_info = 0;
        CUdeviceptr dev_direct_temporal_raygen_record;
        CUdeviceptr dev_direct_spatial_raygen_record;

        // Event handling resources
        bool manual_reset = false;
        std::queue <vk::Extent2D> resize_queue;

        // Options
        bool temporal_reuse = true;
        bool spatial_reuse = true;
	bool brute_force = false;

        // Constructor
        Mamba(const OptixDeviceContext &);

        void render(EditorViewport *, const RenderInfo &, const std::vector <Entity> &, const MaterialDaemon *);
};