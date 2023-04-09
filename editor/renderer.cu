#include "common.hpp"
        
EditorRenderer::EditorRenderer(const Context &context)
                : phdev(context.phdev),
                device(context.device),
                descriptor_pool(context.descriptor_pool),
                command_pool(context.command_pool),
                texture_loader(context.texture_loader),
                extent(context.extent)
{
        resize(context.extent);
        configure_gbuffer_pipeline(context.extent);
        configure_present_pipeline(context.swapchain_format, context.extent);

        // Sobel filter compute shader
        ShaderProgram sobel_compute_shader {
                sobel_comp_shader,
                vk::ShaderStageFlagBits::eCompute
        };

        vk::raii::ShaderModule sobel_compute_module = *sobel_compute_shader.compile(*device);

        // Create pipeline layout
        sobel_dsl = make_descriptor_set_layout(*device, sobel_bindings);

        sobel_ppl = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *sobel_dsl, {}
                }
        };

        // Create pipeline
        sobel_pipeline = vk::raii::Pipeline {
                *device,
                nullptr,
                vk::ComputePipelineCreateInfo {
                        vk::PipelineCreateFlags {},
                        vk::PipelineShaderStageCreateInfo {
                                vk::PipelineShaderStageCreateFlags {},
                                vk::ShaderStageFlagBits::eCompute,
                                *sobel_compute_module,
                                "main"
                        },
                        *sobel_ppl,
                        nullptr
                }
        };

        // Create image and descriptor set for sobel output
        sobel_output = ImageData {
                *phdev, *device,
                vk::Format::eR32Sfloat,
                vk::Extent2D { extent.width, extent.height},
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::ImageAspectFlagBits::eColor
        };

        sobel_dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *sobel_dsl
                }
        }.front());

        // Bind image to descriptor set
        std::array <vk::DescriptorImageInfo, 2> sobel_dset_image_infos {
                vk::DescriptorImageInfo {
                        nullptr,
                        *framebuffer_images.normal.view,
                        vk::ImageLayout::eGeneral
                },

                vk::DescriptorImageInfo {
                        nullptr,
                        *sobel_output.view,
                        vk::ImageLayout::eGeneral
                },
        };

        std::array <vk::WriteDescriptorSet, 2> sobel_dset_writes {
                vk::WriteDescriptorSet {
                        *sobel_dset,
                        0, 0,
                        vk::DescriptorType::eStorageImage,
                        sobel_dset_image_infos[0],
                },

                vk::WriteDescriptorSet {
                        *sobel_dset,
                        1, 0,
                        vk::DescriptorType::eStorageImage,
                        sobel_dset_image_infos[1],
                },
        };

        device->updateDescriptorSets(sobel_dset_writes, nullptr);

        // Create a sampler for the sobel output
        sobel_output_sampler = make_sampler(*device, sobel_output);
        
        bind_ds(*device, present_dset, sobel_output_sampler, sobel_output, 3);
}

void EditorRenderer::configure_gbuffer_pipeline(const vk::Extent2D &extent)
{
        // G-buffer render pass configuration
        std::vector <vk::Format> attachment_formats {
                framebuffer_images.position.format,
                framebuffer_images.normal.format,
                framebuffer_images.material_index.format,
        };

        gbuffer_rp = make_render_pass(*device,
                attachment_formats,
                {
                        vk::AttachmentLoadOp::eClear,
                        vk::AttachmentLoadOp::eClear,
                        vk::AttachmentLoadOp::eClear,
                },
                depth_buffer.format,
                vk::AttachmentLoadOp::eClear
        );

        // Framebuffer
        std::vector <vk::ImageView> attachment_views {
                *framebuffer_images.position.view,
                *framebuffer_images.normal.view,
                *framebuffer_images.material_index.view,
                *depth_buffer.view
        };

        vk::FramebufferCreateInfo fb_info {
                vk::FramebufferCreateFlags {},
                *gbuffer_rp,
                attachment_views,
                extent.width, extent.height, 1
        };

        gbuffer_fb = vk::raii::Framebuffer {*device, fb_info};

        // G-buffer pipeline
        gbuffer_dsl = make_descriptor_set_layout(*device, gbuffer_bindings);

        vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eVertex,
                0, sizeof(PushConstants)
        };

        gbuffer_ppl = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *gbuffer_dsl,
                        push_constant_range
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram gbuffer_vertex {
                gbuffer_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram gbuffer_fragment {
                gbuffer_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo gbuffer_grp_info {
                *device, gbuffer_rp,
                nullptr, nullptr,
                nullptr, nullptr,
                Vertex::vertex_binding(),
                Vertex::vertex_attributes(),
                gbuffer_ppl
        };

        gbuffer_grp_info.vertex_shader = std::move(*gbuffer_vertex.compile(*device));
        gbuffer_grp_info.fragment_shader = std::move(*gbuffer_fragment.compile(*device));
        gbuffer_grp_info.blend_attachments = { true, true, false };
        gbuffer_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        gbuffer_pipeline = make_graphics_pipeline(gbuffer_grp_info);
}

void EditorRenderer::configure_present_pipeline(const vk::Format &swapchain_format, const vk::Extent2D &extent)
{
        // Present render pass configuration
        present_rp = make_render_pass(*device,
                { swapchain_format },
                { vk::AttachmentLoadOp::eClear },
                vk::Format::eD32Sfloat, // TODO: remove the necessity
                                      // for framebuffer... (assert # of
                                      // attachements is 1)
                vk::AttachmentLoadOp::eClear
        );

        // Present pipeline
        present_dsl = make_descriptor_set_layout(*device, present_bindings);

        vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eFragment,
                0, sizeof(PushConstants)
        };

        present_ppl = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *present_dsl,
                        push_constant_range
                }
        };

        // Vertex format
        struct PresentVertex {
                glm::vec3 position;
                glm::vec2 texcoord;
        };

        vk::VertexInputBindingDescription vertex_binding {
                0, sizeof(PresentVertex),
                vk::VertexInputRate::eVertex
        };

        std::vector <vk::VertexInputAttributeDescription> vertex_attributes {
                vk::VertexInputAttributeDescription {
                        0, 0, vk::Format::eR32G32B32Sfloat,
                        offsetof(PresentVertex, position)
                },
                vk::VertexInputAttributeDescription {
                        1, 0, vk::Format::eR32G32Sfloat,
                        offsetof(PresentVertex, texcoord)
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram present_vertex {
                present_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram present_fragment {
                present_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo present_grp_info {
                *device, present_rp,
                nullptr, nullptr,
                nullptr, nullptr,
                vertex_binding,
                vertex_attributes,
                present_ppl
        };

        present_grp_info.vertex_shader = std::move(*present_vertex.compile(*device));
        present_grp_info.fragment_shader = std::move(*present_fragment.compile(*device));
        // present_grp_info.vertex_shader = make_shader_module(*device, KOBRA_SHADERS_DIR "/editor_renderer.glsl");
        present_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        present_pipeline = make_graphics_pipeline(present_grp_info);

        // Allocate buffer resources
        std::vector <PresentVertex> vertices {
                // Triangle 1
                { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
                { {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } },
                { {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },

                // Triangle 2
                { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
                { {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },
                { { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f } }
        };

        present_mesh = BufferData {
                *phdev, *device,
                vertices.size() * sizeof(PresentVertex),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible
                        | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        present_mesh.upload(vertices);

        // Configure descriptor set
        present_dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *present_dsl
                }
        }.front());

        framebuffer_images.position_sampler = make_sampler(*device, framebuffer_images.position);
        framebuffer_images.normal_sampler = make_sampler(*device, framebuffer_images.normal);
        // framebuffer_images.material_index_sampler = make_sampler(*device, framebuffer_images.material_index);

        framebuffer_images.material_index_sampler = vk::raii::Sampler {
                *device,
                vk::SamplerCreateInfo {
                        vk::SamplerCreateFlags {},
                        vk::Filter::eNearest,
                        vk::Filter::eNearest,
                        vk::SamplerMipmapMode::eNearest,
                        vk::SamplerAddressMode::eClampToEdge,
                        vk::SamplerAddressMode::eClampToEdge,
                        vk::SamplerAddressMode::eClampToEdge,
                        0.0f, VK_FALSE, 1.0f,
                        VK_FALSE, vk::CompareOp::eNever,
                        0.0f, 0.0f,
                        vk::BorderColor::eFloatOpaqueWhite,
                        VK_FALSE
                }
        };

        bind_ds(*device, present_dset, framebuffer_images.position_sampler, framebuffer_images.position, 0);
        bind_ds(*device, present_dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 1);
        bind_ds(*device, present_dset, framebuffer_images.material_index_sampler, framebuffer_images.material_index, 2);
}
        
void EditorRenderer::resize(const vk::Extent2D &extent)
{
        static vk::Format formats[] = {
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR32Sint // TODO: increase # of components to add
                // more indices...
        };

        // Other image propreties
        static vk::MemoryPropertyFlags mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
        static vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
        static vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment
                | vk::ImageUsageFlagBits::eTransferSrc
                | vk::ImageUsageFlagBits::eStorage;

        framebuffer_images.position = ImageData {
                *phdev, *device,
                formats[0], extent, tiling,
                usage, mem_flags, aspect
        };

        framebuffer_images.normal = ImageData {
                *phdev, *device,
                formats[1], extent, tiling,
                usage, mem_flags, aspect
        };

        framebuffer_images.material_index = ImageData {
                *phdev, *device,
                formats[2], extent, tiling,
                usage, mem_flags, aspect
        };

        /* Transition images to the correct layout
        vk::raii::Queue queue {*device, 0, 0};
        submit_now(*device, queue, *command_pool,
                   [&](const vk::raii::CommandBuffer &cmd) {
                        framebuffer_images.position.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
                        framebuffer_images.normal.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
                        framebuffer_images.material_index.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
                   }
        ); */

        depth_buffer = DepthBuffer {
                *phdev, *device,
                vk::Format::eD32Sfloat, extent
        };
}

void EditorRenderer::render_gbuffer(const RenderInfo &render_info, const std::vector <Entity> &entities)
{
        // The given entities are assumed to have all the necessary
        // components (Transform and Renderable)

        for (const auto &entity : entities) {
                int id = entity.id;
                const auto &renderable = entity.get <Renderable> ();

                // Allocat descriptor sets if needed
                const auto &meshes = renderable.mesh->submeshes;

                for (int i = 0; i < meshes.size(); i++) {
                        MeshIndex index { id, i };

                        if (gbuffer_dsets_refs.find(index) == gbuffer_dsets_refs.end()) {
                                std::array <vk::DescriptorSetLayout, 1> dsls { *gbuffer_dsl };
                                
                                int new_index = gbuffer_dsets.size();
                                gbuffer_dsets_refs[index] = new_index;
                                
                                gbuffer_dsets.emplace_back(
                                        std::move(
                                                vk::raii::DescriptorSets {
                                                        *device,
                                                        { **descriptor_pool, dsls }
                                                }.front()
                                        )
                                );

                                // Bind inforation to descriptor set
                                Material material = Material::all[meshes[i].material_index];
        
                                std::string albedo = "blank";
                                if (material.has_albedo())
                                        albedo = material.albedo_texture;

                                std::string normal = "blank";
                                if (material.has_normal())
                                        normal = material.normal_texture;

                                const ImageData &albedo_texture = texture_loader->load_texture(albedo);
                                const ImageData &normal_texture = texture_loader->load_texture(normal);

                                vk::raii::Sampler &albedo_sampler = texture_loader->load_sampler(albedo);
                                vk::raii::Sampler &normal_sampler = texture_loader->load_sampler(normal);

                                bind_ds(*device, gbuffer_dsets[new_index], albedo_sampler, albedo_texture, 0);
                                bind_ds(*device, gbuffer_dsets[new_index], normal_sampler, normal_texture, 1);
                        }
                }
        }

        // Render to G-buffer
        // render_info.render_area.apply(render_info.cmd, render_info.extent);
        RenderArea::full().apply(render_info.cmd, extent);

        std::vector <vk::ClearValue> clear_values {
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 0.0f } },
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 0.0f } },
                vk::ClearColorValue { std::array <int32_t, 4> { -1 }},
                vk::ClearDepthStencilValue { 1.0f, 0 }
        };

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *gbuffer_rp,
                        *gbuffer_fb,
                        vk::Rect2D { vk::Offset2D {}, extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *gbuffer_pipeline);

        PushConstants push_constants;

        push_constants.view = render_info.camera.view_matrix(render_info.camera_transform);
        push_constants.projection = render_info.camera.perspective_matrix();

        for (auto &entity : entities) {
                int id = entity.id;
                const auto &transform = entity.get <Transform> ();
                const auto &renderable = entity.get <Renderable> ();
                const auto &meshes = renderable.mesh->submeshes;

                for (int i = 0; i < meshes.size(); i++) {
                        MeshIndex index { id, i };

                        // Bind descriptor set
                        int dset_index = gbuffer_dsets_refs[index];
                        render_info.cmd.bindDescriptorSets(
                                vk::PipelineBindPoint::eGraphics,
                                *gbuffer_ppl,
                                0, { *gbuffer_dsets[dset_index] }, {}
                        );

                        // Bind push constants
                        push_constants.model = transform.matrix();
                        push_constants.material_index = meshes[i].material_index;

                        render_info.cmd.pushConstants <PushConstants> (
                                *gbuffer_ppl,
                                vk::ShaderStageFlagBits::eVertex,
                                0, push_constants
                        );

                        // Draw
                        render_info.cmd.bindVertexBuffers(0, { *renderable.get_vertex_buffer(i).buffer }, { 0 });
                        render_info.cmd.bindIndexBuffer(*renderable.get_index_buffer(i).buffer, 0, vk::IndexType::eUint32);
                        render_info.cmd.drawIndexed(renderable.get_index_count(i), 1, 0, 0, 0);
                }
        }

        render_info.cmd.endRenderPass();

        // Transition layout for normal map
        framebuffer_images.normal.layout = vk::ImageLayout::ePresentSrcKHR;
        framebuffer_images.normal.transition_layout(render_info.cmd, vk::ImageLayout::eGeneral);

        sobel_output.transition_layout(render_info.cmd, vk::ImageLayout::eGeneral);

        // Now apply Sobel filterto get edges
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *sobel_pipeline);
        render_info.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                *sobel_ppl,
                0, { *sobel_dset }, {}
        );

        render_info.cmd.dispatch(
                (extent.width + 15) / 16,
                (extent.height + 15) / 16,
                1
        );
}

void EditorRenderer::render_present(const RenderInfo &render_info)
{
        // TODO: pass different render options

        // Render to screen
        render_info.render_area.apply(render_info.cmd, render_info.extent);

        // WARNING: there is an issue where the alpha channel is not being written to the framebuffer
        // the current workaround is to use a clear value of 1.0f for the alpha channel
        std::vector <vk::ClearValue> clear_values {
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } },
                vk::ClearDepthStencilValue { 1.0f, 0 }
        };

        // TODO: remove depht buffer from this...

        // Transition framebuffer images to shader read
        // TODO: some way to avoid this hackery
        framebuffer_images.position.layout = vk::ImageLayout::ePresentSrcKHR;
        framebuffer_images.normal.layout = vk::ImageLayout::eGeneral;
        framebuffer_images.material_index.layout = vk::ImageLayout::ePresentSrcKHR;

        framebuffer_images.position.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        framebuffer_images.normal.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        framebuffer_images.material_index.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        sobel_output.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *present_rp,
                        *render_info.framebuffer,
                        vk::Rect2D { vk::Offset2D {}, render_info.extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *present_pipeline);

        // Bind descriptor set
        render_info.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *present_ppl,
                0, { *present_dset }, {}
        );

        // Draw
        render_info.cmd.bindVertexBuffers(0, { *present_mesh.buffer }, { 0 });
        render_info.cmd.draw(6, 1, 0, 0);

        render_info.cmd.endRenderPass();
}
