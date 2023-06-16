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
        OptixPipeline secondary_ppl = 0;

        OptixProgramGroup raygen_direct_primary = 0;
        OptixProgramGroup raygen_direct_temporal = 0;
        OptixProgramGroup raygen_direct_spatial = 0;
        
	OptixProgramGroup raygen_secondary = 0;

        OptixProgramGroup closest_hit = 0;
        OptixProgramGroup miss = 0;

        OptixShaderBindingTable direct_initial_sbt = {};
        OptixShaderBindingTable direct_temporal_sbt = {};
        OptixShaderBindingTable direct_spatial_sbt = {};

	OptixShaderBindingTable secondary_sbt = {};

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
	// TODO: wrapper struct
        bool temporal_reuse = false;
        bool spatial_reuse = false;
	
	struct {
		bool direct_lighting = false;
		bool indirect_lighting = false;
		bool irradiance = true;
	
		bool render_probes = false;
		bool render_probe_radiance = false;
	} options;

        // Constructor
        Mamba(const OptixDeviceContext &);

        void render(EditorViewport *, const RenderInfo &, const std::vector <Entity> &, const MaterialDaemon *);
};
