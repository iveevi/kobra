#pragma once

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/common.hpp"
#include "include/daemons/mesh.hpp"
#include "include/daemons/transform.hpp"
#include "include/engine/irradiance_computer.hpp"
#include "include/layers/common.hpp"
#include "include/layers/forward_renderer.hpp"
#include "include/layers/framer.hpp"
#include "include/layers/image_renderer.hpp"
#include "include/layers/ui.hpp"
#include "include/project.hpp"
#include "include/scene.hpp"
#include "include/shader_program.hpp"
#include "include/ui/attachment.hpp"
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

struct WindowBounds {
        glm::vec2 min;
        glm::vec2 max;
};

inline bool within(const glm::vec2 &x, const WindowBounds &wb)
{
        return x.x >= wb.min.x && x.x <= wb.max.x
                && x.y >= wb.min.y && x.y <= wb.max.y;
}

inline glm::vec2 normalize(const glm::vec2 &x, const WindowBounds &wb)
{
        return glm::vec2 {
                (x.x - wb.min.x) / (wb.max.x - wb.min.x),
                (x.y - wb.min.y) / (wb.max.y - wb.min.y),
        };
}

// Input event handling
struct InputRequest {
        glm::vec2 position;
        glm::vec2 delta;
        glm::vec2 start;
        
        enum {
                eNone,
                ePress,
                eRelease,
                eDrag
        } type;
};

struct InputContext {
        // Viewport window
        struct : WindowBounds {
                float sensitivity = 0.001f;
                float yaw = 0.0f;
                float pitch = 0.0f;
                bool dragging = false;
        } viewport;

        // Material preview window
        struct : WindowBounds {
                float sensitivity = 0.05f;
                bool dragging = false;
        } material_preview;

        // Input requests
        std::queue <InputRequest> requests;

        // Dragging states
        bool dragging = false;
        bool alt_dragging = false;
        glm::vec2 drag_start;
};

extern InputContext input_context; // TODO: g_input_context

// Render packet information
struct RenderInfo {
        Camera camera;
        Transform camera_transform;
        bool camera_transform_dirty = true;

        RenderArea render_area = RenderArea::full();
        std::set <int> highlighted_entities;
        vk::Extent2D extent;
        const vk::raii::CommandBuffer &cmd = nullptr;

        // TODO: attach entities and transform daemon, etc..

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
                ePathTracer,
                // ePathTraced_Amadeus
        } mode = ePathTracer;

        enum {
                eRasterized,
                eRaytraced
        } backend = eRaytraced;

        bool bounding_boxes = false;
        bool initialized = false;
        bool sparse_gi_reset = false;
        bool path_tracer_reset = false;
};

// Modules within the editor rendering pipeline
struct SimplePipeline {
        vk::raii::PipelineLayout pipeline_layout = nullptr;
        vk::raii::Pipeline pipeline = nullptr;

        vk::raii::DescriptorSetLayout dsl = nullptr;
        vk::raii::DescriptorSet dset = nullptr;
};

void set_imgui_theme();
