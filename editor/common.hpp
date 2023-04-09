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

/* Rasterizing for primary intersection and G-buffer
struct PrimaryRasterizer {

}; */

// Push constants for editor renderer
struct PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	int material_index;
};

// Static member variables
static const std::vector <DescriptorSetLayoutBinding> gbuffer_bindings {
        // Albedo for opacity masking
	DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

        // Normal map
	DescriptorSetLayoutBinding {
		1, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

static const std::vector <DescriptorSetLayoutBinding> present_bindings {
        // Position
	DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

        // Normal
        DescriptorSetLayoutBinding {
		1, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

        // Material index
        DescriptorSetLayoutBinding {
		2, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

        // Extra
        // TODO: tabular system for this and the shaders...
        DescriptorSetLayoutBinding {
                3, vk::DescriptorType::eCombinedImageSampler,
                1, vk::ShaderStageFlagBits::eFragment
        },
};

static const std::vector <DescriptorSetLayoutBinding> sobel_bindings {
        // Normal
        DescriptorSetLayoutBinding {
                0, vk::DescriptorType::eStorageImage,
                1, vk::ShaderStageFlagBits::eCompute
        },

        // Output
        DescriptorSetLayoutBinding {
                1, vk::DescriptorType::eStorageImage,
                1, vk::ShaderStageFlagBits::eCompute
        },
};

// Editor renderer shaders
extern const char *gbuffer_vert_shader;
extern const char *gbuffer_frag_shader;

// TODO: use a simple mesh shader for this...
extern const char *present_vert_shader;
extern const char *present_frag_shader;
extern const char *sobel_comp_shader;

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

        // Pipelines and render passes
        vk::raii::RenderPass gbuffer_rp = nullptr;
        vk::raii::PipelineLayout gbuffer_ppl = nullptr;
        vk::raii::Pipeline gbuffer_pipeline = nullptr;

        vk::raii::RenderPass present_rp = nullptr;
        vk::raii::PipelineLayout present_ppl = nullptr;
        vk::raii::Pipeline present_pipeline = nullptr;

        vk::raii::PipelineLayout sobel_ppl = nullptr;
        vk::raii::Pipeline sobel_pipeline = nullptr;

        // Descriptor sets and layouts
        vk::raii::DescriptorSetLayout gbuffer_dsl = nullptr;
        vk::raii::DescriptorSetLayout present_dsl = nullptr;
        vk::raii::DescriptorSetLayout sobel_dsl = nullptr;

        vk::raii::DescriptorSet present_dset = nullptr;
        vk::raii::DescriptorSet sobel_dset = nullptr;
        
        using MeshIndex = std::pair <int, int>; // Entity, mesh index
        std::map <MeshIndex, int> gbuffer_dsets_refs;
        std::vector <vk::raii::DescriptorSet> gbuffer_dsets;

        // Additional resources
        BufferData present_mesh = nullptr;

        ImageData sobel_output = nullptr;
        vk::raii::Sampler sobel_output_sampler = nullptr;

        // Current viewport extent
        vk::Extent2D extent;

        EditorRenderer() = delete;
        EditorRenderer(const Context &);

        void configure_gbuffer_pipeline(const vk::Extent2D &);
        void configure_present_pipeline(const vk::Format &, const vk::Extent2D &);

        void resize(const vk::Extent2D &);

        void render_gbuffer(const RenderInfo &, const std::vector <Entity> &);
        void render_present(const RenderInfo &);

};
