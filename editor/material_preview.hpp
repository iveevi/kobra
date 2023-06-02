#pragma once

// Standard headers
#include <filesystem>

// Engine headers
#include "api.hpp"
#include "include/backend.hpp"
#include "include/material.hpp"
#include "include/shader_program.hpp"
#include "include/transform.hpp"
#include "include/daemons/material.hpp"

struct PushConstants {
        // Camera location
        alignas(16) glm::vec3 origin;

        // Material properties
        alignas(16) glm::vec3 diffuse;
        alignas(16) glm::vec3 specular;
        float roughness;
};

struct MaterialPreview {
        // Vulkan structures
        vk::Device device;
        vk::PhysicalDevice phdev;
        vk::CommandPool command_pool;
        vk::DescriptorPool descriptor_pool;

        vk::Pipeline pipeline;
        vk::PipelineLayout pipeline_layout;

        vk::DescriptorSetLayout descriptor_set_layout;
        vk::DescriptorSet descriptor_set;

        // Target image
        api::Image display;

        // Environment image
        api::Image environment;
        vk::Sampler sampler;

        // Camera and material
        glm::vec3 origin;
        int32_t index;
};

void load_environment_map(MaterialPreview *mp, const std::filesystem::path &);
MaterialPreview *make_material_preview(const kobra::Context &);
void destroy_material_preview(MaterialPreview *);
void render_material_preview(const vk::CommandBuffer &, MaterialPreview *, const kobra::MaterialDaemon *);
