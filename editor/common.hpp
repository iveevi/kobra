#pragma once

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/common.hpp"
#include "include/layers/common.hpp"
#include "include/layers/forward_renderer.hpp"
#include "include/layers/image_renderer.hpp"
#include "include/layers/objectifier.hpp"
#include "include/layers/ui.hpp"
#include "include/project.hpp"
#include "include/scene.hpp"
#include "include/shader_program.hpp"
#include "include/ui/attachment.hpp"
#include "include/engine/irradiance_computer.hpp"
#include "include/amadeus/armada.cuh"
#include "include/amadeus/path_tracer.cuh"
#include "include/amadeus/restir.cuh"
#include "include/layers/framer.hpp"
#include "include/cuda/color.cuh"
#include "include/layers/denoiser.cuh"
#include "include/daemons/transform.hpp"
#include "include/vertex.hpp"

// Native File Dialog
#include <nfd.h>

// ImPlot headers
#include <implot/implot.h>
#include <implot/implot_internal.h>

// ImGuizmo
#include <ImGuizmo/ImGuizmo.h>

// Extra GLM headers
#include <glm/gtc/type_ptr.hpp>

// Aliasing declarations
using namespace kobra;

// Global communications structure
struct Application {
	Context context;
	float speed = 10.0f;
};

// Render packet information
struct RenderInfo {
        const Camera &camera;
        const Transform &camera_transform;
        const vk::raii::CommandBuffer &cmd;
        const vk::raii::Framebuffer &framebuffer;
        const vk::Extent2D &extent;
        const RenderArea &render_area = RenderArea::full();
};

// Editor rendering
struct EditorRenderer {
        // Vulkan structures
        const vk::raii::Device *device = nullptr;
        const vk::raii::PhysicalDevice *phdev = nullptr;
        const vk::raii::DescriptorPool *descriptor_pool = nullptr;
        const vk::raii::CommandPool *command_pool = nullptr;
        TextureLoader *texture_loader = nullptr;

        // Buffers
        struct framebuffer_images {
                ImageData position = nullptr;
                ImageData normal = nullptr;
                ImageData material_index = nullptr;

                vk::raii::Sampler position_sampler = nullptr;
                vk::raii::Sampler normal_sampler = nullptr;
                vk::raii::Sampler material_index_sampler = nullptr;
        } framebuffer_images;

        DepthBuffer depth_buffer = nullptr;

        vk::raii::Framebuffer gbuffer_fb = nullptr;

        // Pipeline resources
        using MeshIndex = std::pair <int, int>; // Entity, mesh index
        
        struct {
                vk::raii::RenderPass render_pass = nullptr;
                vk::raii::PipelineLayout pipeline_layout = nullptr;
                vk::raii::Pipeline pipeline = nullptr;
       
                vk::raii::DescriptorSetLayout dsl = nullptr;
                std::map <MeshIndex, int> dset_refs;
                std::vector <vk::raii::DescriptorSet> dsets;
        } gbuffer;

        struct {
                vk::raii::RenderPass render_pass = nullptr;
                vk::raii::PipelineLayout pipeline_layout = nullptr;
                vk::raii::Pipeline pipeline = nullptr;
        
                vk::raii::DescriptorSetLayout dsl = nullptr;
                std::map <MeshIndex, int> dset_refs;
                std::vector <vk::raii::DescriptorSet> dsets;
        } albedo;

        struct {
                vk::raii::RenderPass render_pass = nullptr;
                vk::raii::PipelineLayout pipeline_layout = nullptr;
                vk::raii::Pipeline pipeline = nullptr;
        
                vk::raii::DescriptorSetLayout dsl = nullptr;
                vk::raii::DescriptorSet dset = nullptr;
        
                BufferData mesh = nullptr;
        } triangulation;

        struct {
                vk::raii::PipelineLayout pipeline_layout = nullptr;
                vk::raii::Pipeline pipeline = nullptr;
        
                vk::raii::DescriptorSetLayout dsl = nullptr;
                vk::raii::DescriptorSet dset = nullptr;
                
                ImageData output = nullptr;
                vk::raii::Sampler output_sampler = nullptr;
        } sobel;

        // Current viewport extent
        vk::Extent2D extent;

        // Rendering mode and parameters
        struct RenderState {
                enum {
                        eTriangulation,
                        eWireframe,
                        eNormals,
                        eAlbedo,
                        eSparseRTX
                } mode = eTriangulation;

                bool bounding_boxes = false;
        } render_state;

        // TODO: table mapping render_state to function for presenting

        EditorRenderer() = delete;
        EditorRenderer(const Context &);

        void configure_gbuffer_pipeline();
        void configure_albedo_pipeline(const vk::Format &);
        void configure_triangulation_pipeline(const vk::Format &);
        void configure_sobel_pipeline();

        void resize(const vk::Extent2D &);

        void render_gbuffer(const RenderInfo &, const std::vector <Entity> &);
        void render_albedo(const RenderInfo &, const std::vector <Entity> &);
        void render_triangulation(const RenderInfo &);

        void render(const RenderInfo &, const std::vector <Entity> &);

        // ImGui memu
        void menu();

};
