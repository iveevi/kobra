#pragma once

// Engine headers
#include "include/optix/core.cuh"

// Local headers
#include "editor/optix/mamba_shader.cuh"

// Mamba global illumination
struct Mamba {
        // CUDA resources
        CUstream stream_direct = 0;
        CUstream stream_indirect = 0;

        // OptiX resources
        OptixPipeline pipeline = 0;
        OptixModule module = 0;
        
        OptixProgramGroup ray_generation = 0;
        OptixProgramGroup closest_hit = 0;
        OptixProgramGroup miss = 0;

        OptixShaderBindingTable sbt = {};

        // Constructor
        Mamba(const OptixDeviceContext &);
};
