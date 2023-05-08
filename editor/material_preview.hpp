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

void load_environment_map(MaterialPreview *mp, const std::filesystem::path &path)
{
        // Load texture
        kobra::RawImage image = kobra::load_texture(path);

        // Create image
        vk::PhysicalDeviceMemoryProperties mem_props = mp->phdev.getMemoryProperties();
        
        vk::Format format = (image.type == kobra::RawImage::RGBA_32_F) ?
                vk::Format::eR32G32B32A32Sfloat : vk::Format::eR8G8B8A8Unorm;

        mp->environment = api::make_image(mp->device, {
                image.width, image.height, format,
                vk::ImageUsageFlagBits::eSampled
                        | vk::ImageUsageFlagBits::eTransferDst
                        | vk::ImageUsageFlagBits::eTransferSrc
        }, mem_props);

        // Copy data
        vk::CommandBufferAllocateInfo cmd_alloc_info {
                mp->command_pool, vk::CommandBufferLevel::ePrimary, 1
        };

        vk::CommandBuffer cmd = mp->device.allocateCommandBuffers(cmd_alloc_info).front();

        vk::CommandBufferBeginInfo cmd_begin_info {
                vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        cmd.begin(cmd_begin_info);

        vk::ImageSubresourceRange subresource_range {
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1
        };

        vk::ImageMemoryBarrier barrier {
                vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eTransferWrite,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                mp->environment.image, subresource_range
        };

        cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlags {}, {}, {}, barrier
        );

        std::cout << "ALLOCATING BUFFER of size " << mp->environment.requirements.size << std::endl;
        api::Buffer buffer = api::make_buffer(mp->device, mp->environment.requirements.size, mem_props);
        api::upload(mp->device, buffer, image.data);

        vk::BufferImageCopy copy_region {
                0, 0, 0,
                vk::ImageSubresourceLayers {
                        vk::ImageAspectFlagBits::eColor, 0, 0, 1
                },
                vk::Offset3D { 0, 0, 0 },
                vk::Extent3D { image.width, image.height, 1 }
        };

        cmd.copyBufferToImage(buffer.buffer, mp->environment.image, vk::ImageLayout::eTransferDstOptimal, copy_region);

        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader,
                vk::DependencyFlags {}, {}, {}, barrier
        );

        cmd.end();

        vk::SubmitInfo submit_info {
                {}, {}, cmd, {}
        };

        mp->device.getQueue(0, 0).submit(submit_info, nullptr);
        mp->device.waitIdle();

        destroy_buffer(mp->device, buffer);

        // Create sampler for the environment map
        vk::SamplerCreateInfo sampler_info {
                {}, vk::Filter::eLinear, vk::Filter::eLinear,
                vk::SamplerMipmapMode::eLinear,
                vk::SamplerAddressMode::eRepeat,
                vk::SamplerAddressMode::eRepeat,
                vk::SamplerAddressMode::eRepeat,
                0.0f, VK_FALSE, 16.0f, VK_FALSE,
                vk::CompareOp::eNever, 0.0f, 0.0f,
                vk::BorderColor::eFloatOpaqueWhite, VK_FALSE
        };

        mp->sampler = mp->device.createSampler(sampler_info);

        std::cout << "LOADING ENVIRONMENT IMAGE: " << image.width << "x" << image.height << "\n";

        // Bind to the descriptor set
        vk::DescriptorImageInfo descriptor_image_info {
                mp->sampler, mp->environment.view,
                vk::ImageLayout::eShaderReadOnlyOptimal
        };

        vk::WriteDescriptorSet write_descriptor_set {
                mp->descriptor_set, 1, 0, 1,
                vk::DescriptorType::eCombinedImageSampler,
                &descriptor_image_info, nullptr, nullptr
        };

        mp->device.updateDescriptorSets(write_descriptor_set, nullptr);

        std::cout << "DONE\n";

        // Free resources
        mp->device.freeCommandBuffers(mp->command_pool, cmd);
}

MaterialPreview *make_material_preview(const kobra::Context &context)
{
        constexpr const char *SHADER_SOURCE = KOBRA_DIR "/editor/shaders/material_preview.glsl";

        constexpr const std::array <vk::DescriptorSetLayoutBinding, 2> BINDINGS = {
                vk::DescriptorSetLayoutBinding {
                        0, vk::DescriptorType::eStorageImage,
                        1, vk::ShaderStageFlagBits::eCompute
                },
                
                vk::DescriptorSetLayoutBinding {
                        1, vk::DescriptorType::eCombinedImageSampler,
                        1, vk::ShaderStageFlagBits::eCompute
                },
        };

        MaterialPreview *mp = new MaterialPreview;
        mp->device = **context.device;
        mp->phdev = **context.phdev;
        mp->command_pool = **context.command_pool;
        mp->descriptor_pool = **context.descriptor_pool;
        mp->index = -1;

        // Compile the shader
        std::string content = kobra::common::read_file(SHADER_SOURCE);
        kobra::ShaderProgram program { content, vk::ShaderStageFlagBits::eCompute };

        auto opt_shader = program.compile(mp->device, {}, { KOBRA_DIR "/editor/shaders" });
        KOBRA_ASSERT(opt_shader.has_value(), "Failed to compile shader");

        // Create the pipeline layout
        vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)
        };

        mp->descriptor_set_layout = mp->device
                .createDescriptorSetLayout({
                        vk::DescriptorSetLayoutCreateFlags {},
                        BINDINGS
                });

        vk::PipelineLayoutCreateInfo pipeline_layout_info {
                {}, mp->descriptor_set_layout, push_constant_range
        };

        mp->pipeline_layout = mp->device
                .createPipelineLayout(pipeline_layout_info);

        // Create the compute pipeline
        vk::PipelineShaderStageCreateInfo shader_stage_info {
                {}, vk::ShaderStageFlagBits::eCompute, *opt_shader, "main"
        };

        vk::ComputePipelineCreateInfo pipeline_info {
                {}, shader_stage_info, mp->pipeline_layout
        };

        auto result = mp->device
                .createComputePipeline({}, pipeline_info);

        KOBRA_ASSERT(result.result == vk::Result::eSuccess, "Failed to create pipeline");

        mp->pipeline = result.value;

        // Allocate target image
        vk::PhysicalDeviceMemoryProperties mem_props = mp->phdev.getMemoryProperties();

        mp->display = api::make_image(mp->device, {
                512, 512, vk::Format::eR32G32B32A32Sfloat,
                vk::ImageUsageFlagBits::eStorage
                        | vk::ImageUsageFlagBits::eSampled,
        }, mem_props);

        // Transition the image to general layout
        vk::CommandBufferAllocateInfo cmd_buffer_info {
                mp->command_pool, vk::CommandBufferLevel::ePrimary, 1
        };

        vk::CommandBuffer cmd = mp->device
                .allocateCommandBuffers(cmd_buffer_info).front();

        cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        api::transition_image_layout(cmd, mp->display, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        cmd.end();

        vk::SubmitInfo submit_info {
                {}, {}, cmd, {}
        };

        mp->device.getQueue(0, 0).submit(submit_info, nullptr);
        mp->device.waitIdle();

        // Allocate a descriptor set
        vk::DescriptorSetAllocateInfo descriptor_set_info {
                mp->descriptor_pool,
                1, &mp->descriptor_set_layout
        };

        mp->descriptor_set = mp->device
                .allocateDescriptorSets(descriptor_set_info).front();

        // Bind the display image to the descriptor set
        vk::DescriptorImageInfo descriptor_image_info {
                {}, mp->display.view, vk::ImageLayout::eGeneral
        };

        vk::WriteDescriptorSet write_descriptor_set {
                mp->descriptor_set, 0, 0, 1,
                vk::DescriptorType::eStorageImage,
                &descriptor_image_info, nullptr, nullptr
        };

        mp->device.updateDescriptorSets(write_descriptor_set, nullptr);

        // Free resources
        mp->device.freeCommandBuffers(mp->command_pool, cmd);
        mp->device.destroyShaderModule(*opt_shader);

        return mp;
}

void destroy_material_preview(MaterialPreview *mp)
{
        mp->device.destroyPipeline(mp->pipeline);
        mp->device.destroyPipelineLayout(mp->pipeline_layout);
        mp->device.destroyDescriptorSetLayout(mp->descriptor_set_layout);

        destroy_image(mp->device, mp->display);
        destroy_image(mp->device, mp->environment);
        
        mp->device.destroySampler(mp->sampler);

        delete mp;
}

// TODO: pass the actual material
void render_material_preview(const vk::CommandBuffer &cmd, MaterialPreview *mp, const kobra::MaterialDaemon *md)
{
        PushConstants push_constants;
        push_constants.origin = mp->origin;

        if (mp->index >= 0) {
                // const kobra::Material &material = kobra::Material::all[mp->index];
                const kobra::Material &material = md->materials[mp->index];
                push_constants.diffuse = material.diffuse;
                push_constants.specular = material.specular;
                push_constants.roughness = material.roughness;
        }

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, mp->pipeline);
        cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                mp->pipeline_layout, 0, mp->descriptor_set, {}
        );

        cmd.pushConstants <PushConstants> (
                mp->pipeline_layout,
                vk::ShaderStageFlagBits::eCompute,
                0, push_constants
        );

        cmd.dispatch(512 / 16, 512 / 16, 1);
}
