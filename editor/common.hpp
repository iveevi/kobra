#pragma once

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/common.hpp"
#include "include/layers/common.hpp"
#include "include/layers/forward_renderer.hpp"
#include "include/layers/image_renderer.hpp"
#include "include/layers/mesh_memory.hpp"
#include "include/layers/objectifier.hpp"
#include "include/layers/ui.hpp"
#include "include/project.hpp"
#include "include/scene.hpp"
#include "include/shader_program.hpp"
#include "include/ui/attachment.hpp"
#include "include/engine/irradiance_computer.hpp"
#include "include/layers/framer.hpp"
#include "include/daemons/transform.hpp"
#include "include/vertex.hpp"

// Check if compiling with CUDA
#ifdef __CUDACC__

// CUDA headers
#include "include/amadeus/armada.cuh"
#include "include/amadeus/path_tracer.cuh"
#include "include/amadeus/restir.cuh"
#include "include/cuda/color.cuh"
#include "include/layers/denoiser.cuh"

// Editor headers
#include "gbuffer_rtx_shader.cuh"
#include "path_tracer.cuh"

#endif

// Native File Dialog
#include <nfd.h>

// ImPlot headers
#include <implot/implot.h>
#include <implot/implot_internal.h>

// ImGuizmo
#include <ImGuizmo/ImGuizmo.h>

// Extra GLM headers
#include <glm/gtc/type_ptr.hpp>

// Editor headers
#include "app.hpp"

// Aliasing declarations
using namespace kobra;

// Render packet information
struct RenderInfo {
        Camera camera;
        RenderArea render_area = RenderArea::full();
        Transform camera_transform;
        std::set <int> highlighted_entities;
        vk::Extent2D extent;
        const vk::raii::CommandBuffer &cmd = nullptr;

        RenderInfo(const vk::raii::CommandBuffer &_cmd) : cmd(_cmd) {}
};

// Menu options
struct MenuOptions {
        Camera *camera = nullptr;
        float *speed = nullptr;
};

// Editor render state
struct RenderState {
        enum {
                eTriangulation,
                eWireframe,
                eNormals,
                eTextureCoordinates,
                eAlbedo,
                eSparseGlobalIllumination,
                ePathTraced,
                // ePathTraced_Amadeus
        } mode = eTriangulation;

        enum {
                eRasterized,
                eRaytraced
        } backend = eRasterized;

        bool bounding_boxes = false;
        bool initialized = false;
};

// Modules within the editor rendering pipeline
struct SimplePipeline {
        vk::raii::PipelineLayout pipeline_layout = nullptr;
        vk::raii::Pipeline pipeline = nullptr;

        vk::raii::DescriptorSetLayout dsl = nullptr;
        vk::raii::DescriptorSet dset = nullptr;
};

void set_imgui_theme();
