#include "common.hpp"

// Editor renderer shaders
extern const char *gbuffer_vert_shader;
extern const char *gbuffer_frag_shader;

extern const char *albedo_vert_shader;
extern const char *albedo_frag_shader;

// TODO: use a simple mesh shader for this...
extern const char *presentation_vert_shader;

extern const char *normal_frag_shader;
extern const char *triangulation_frag_shader;
extern const char *selection_frag_shader;

extern const char *sobel_comp_shader;

// Static variables
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

static const std::vector <DescriptorSetLayoutBinding> albedo_bindings {
        // Albedo
	DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

static const std::vector <DescriptorSetLayoutBinding> normal_bindings {
        // Normal framebuffer
        DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

static const std::vector <DescriptorSetLayoutBinding> triangulation_bindings {
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
        // Position
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

// Push constants for editor renderer
struct GBuffer_PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	int material_index;
        int texture_status;
};

struct Albedo_PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;

        glm::vec4 albedo;
        int has_albedo;
};

// Presentation Vertex format
struct FlatVertex {
        glm::vec3 position;
        glm::vec2 texcoord;
};

static const vk::VertexInputBindingDescription flat_vertex_binding {
        0, sizeof(FlatVertex),
        vk::VertexInputRate::eVertex
};

static const std::vector <vk::VertexInputAttributeDescription> flat_vertex_attributes {
        vk::VertexInputAttributeDescription {
                0, 0, vk::Format::eR32G32B32Sfloat,
                offsetof(FlatVertex, position)
        },
        vk::VertexInputAttributeDescription {
                1, 0, vk::Format::eR32G32Sfloat,
                offsetof(FlatVertex, texcoord)
        }
};
        
// Constructor
EditorRenderer::EditorRenderer(const Context &context)
                : phdev(context.phdev),
                device(context.device),
                descriptor_pool(context.descriptor_pool),
                command_pool(context.command_pool),
                texture_loader(context.texture_loader)
{
        resize(context.extent);
        configure_gbuffer_pipeline();
        configure_albedo_pipeline(context.swapchain_format);
        configure_normals_pipeline(context.swapchain_format);
        configure_triangulation_pipeline(context.swapchain_format);
        configure_sobel_pipeline();

        render_state.initialized = true;
}

void EditorRenderer::configure_gbuffer_pipeline()
{
        // G-buffer render pass configuration
        std::vector <vk::Format> attachment_formats {
                framebuffer_images.position.format,
                framebuffer_images.normal.format,
                framebuffer_images.material_index.format,
        };

        gbuffer.render_pass = make_render_pass(*device,
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
                *gbuffer.render_pass,
                attachment_views,
                extent.width, extent.height, 1
        };

        gbuffer_fb = vk::raii::Framebuffer {*device, fb_info};

        // G-buffer pipeline
        gbuffer.dsl = make_descriptor_set_layout(*device, gbuffer_bindings);

        vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eVertex,
                0, sizeof(GBuffer_PushConstants)
        };

        gbuffer.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *gbuffer.dsl,
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
                *device, gbuffer.render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                Vertex::vertex_binding(),
                Vertex::vertex_attributes(),
                gbuffer.pipeline_layout,
        };

        gbuffer_grp_info.vertex_shader = std::move(*gbuffer_vertex.compile(*device));
        gbuffer_grp_info.fragment_shader = std::move(*gbuffer_fragment.compile(*device));
        gbuffer_grp_info.blend_attachments = { true, true, false };
        gbuffer_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        gbuffer.pipeline = make_graphics_pipeline(gbuffer_grp_info);
}

void EditorRenderer::configure_albedo_pipeline(const vk::Format &swapchain_format)
{
        albedo.render_pass = make_render_pass(*device,
                { swapchain_format },
                { vk::AttachmentLoadOp::eClear },
                vk::Format::eD32Sfloat,
                vk::AttachmentLoadOp::eClear
        );

        // G-buffer pipeline
        albedo.dsl = make_descriptor_set_layout(*device, albedo_bindings);

        vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eVertex,
                0, sizeof(Albedo_PushConstants)
        };

        albedo.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *albedo.dsl,
                        push_constant_range
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram albedo_vertex {
                albedo_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram albedo_fragment {
                albedo_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo albedo_grp_info {
                *device, albedo.render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                Vertex::vertex_binding(),
                Vertex::vertex_attributes(),
                albedo.pipeline_layout,
        };

        albedo_grp_info.vertex_shader = std::move(*albedo_vertex.compile(*device));
        albedo_grp_info.fragment_shader = std::move(*albedo_fragment.compile(*device));
        albedo_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        albedo.pipeline = make_graphics_pipeline(albedo_grp_info);
}

void EditorRenderer::configure_normals_pipeline(const vk::Format &swapchain_format)
{
        normal.render_pass = make_render_pass(*device,
                { swapchain_format },
                { vk::AttachmentLoadOp::eClear },
                vk::Format::eD32Sfloat,
                vk::AttachmentLoadOp::eClear
        );

        normal.dsl = make_descriptor_set_layout(*device, normal_bindings);

        normal.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *normal.dsl, {}
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram normal_vertex {
                presentation_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram normal_fragment {
                normal_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo normal_grp_info {
                *device, normal.render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                flat_vertex_binding,
                flat_vertex_attributes,
                normal.pipeline_layout,
        };

        normal_grp_info.vertex_shader = std::move(*normal_vertex.compile(*device));
        normal_grp_info.fragment_shader = std::move(*normal_fragment.compile(*device));
        normal_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        normal.pipeline = make_graphics_pipeline(normal_grp_info);
        
        // Configure descriptor set
        normal.dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *normal.dsl
                }
        }.front());
        
        bind_ds(*device, normal.dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 0);
}

void EditorRenderer::configure_triangulation_pipeline(const vk::Format &swapchain_format)
{
        // triangulation render pass configuration
        triangulation.render_pass = make_render_pass(*device,
                { swapchain_format },
                { vk::AttachmentLoadOp::eClear },
                vk::Format::eD32Sfloat, // TODO: remove the necessity
                                      // for framebuffer... (assert # of
                                      // attachements is 1)
                vk::AttachmentLoadOp::eClear
        );

        // triangulation pipeline
        triangulation.dsl = make_descriptor_set_layout(*device, triangulation_bindings);

        /* vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eFragment,
                0, sizeof(GBuffer_PushConstants)
        }; */

        triangulation.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *triangulation.dsl,
                        nullptr
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram triangulation_vertex {
                presentation_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram triangulation_fragment {
                triangulation_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo triangulation_grp_info {
                *device, triangulation.render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                flat_vertex_binding,
                flat_vertex_attributes,
                triangulation.pipeline_layout
        };

        triangulation_grp_info.vertex_shader = std::move(*triangulation_vertex.compile(*device));
        triangulation_grp_info.fragment_shader = std::move(*triangulation_fragment.compile(*device));
        // triangulation_grp_info.vertex_shader = make_shader_module(*device, KOBRA_SHADERS_DIR "/editor_renderer.glsl");
        triangulation_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        triangulation.pipeline = make_graphics_pipeline(triangulation_grp_info);

        // Allocate buffer resources
        std::vector <FlatVertex> vertices {
                // Triangle 1
                { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
                { {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } },
                { {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },

                // Triangle 2
                { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
                { {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },
                { { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f } }
        };

        // TODO: keep somewhere else
        presentation_mesh = BufferData {
                *phdev, *device,
                vertices.size() * sizeof(FlatVertex),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible
                        | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        presentation_mesh.upload(vertices);

        // Configure descriptor set
        triangulation.dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *triangulation.dsl
                }
        }.front());

        bind_ds(*device, triangulation.dset, framebuffer_images.position_sampler, framebuffer_images.position, 0);
        bind_ds(*device, triangulation.dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 1);
        bind_ds(*device, triangulation.dset, framebuffer_images.material_index_sampler, framebuffer_images.material_index, 2);
}

void EditorRenderer::configure_sobel_pipeline()
{
        // Sobel filter compute shader
        ShaderProgram sobel_compute_shader {
                sobel_comp_shader,
                vk::ShaderStageFlagBits::eCompute
        };

        vk::raii::ShaderModule sobel_compute_module = *sobel_compute_shader.compile(*device);

        // Create pipeline layout
        sobel.dsl = make_descriptor_set_layout(*device, sobel_bindings);

        sobel.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *sobel.dsl, {}
                }
        };

        // Create pipeline
        sobel.pipeline = vk::raii::Pipeline {
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
                        *sobel.pipeline_layout,
                        nullptr
                }
        };

        // Create image and descriptor set for sobel output
        sobel.dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *sobel.dsl
                }
        }.front());

        // Bind image to descriptor set
        std::array <vk::DescriptorImageInfo, 2> sobel_dset_image_infos {
                vk::DescriptorImageInfo {
                        nullptr,
                        *framebuffer_images.material_index.view,
                        vk::ImageLayout::eGeneral
                },

                vk::DescriptorImageInfo {
                        nullptr,
                        *sobel.output.view,
                        vk::ImageLayout::eGeneral
                },
        };

        std::array <vk::WriteDescriptorSet, 2> sobel_dset_writes {
                vk::WriteDescriptorSet {
                        *sobel.dset,
                        0, 0,
                        vk::DescriptorType::eStorageImage,
                        sobel_dset_image_infos[0],
                },

                vk::WriteDescriptorSet {
                        *sobel.dset,
                        1, 0,
                        vk::DescriptorType::eStorageImage,
                        sobel_dset_image_infos[1],
                },
        };

        device->updateDescriptorSets(sobel_dset_writes, nullptr);

        // Create a sampler for the sobel output
        sobel.output_sampler = make_sampler(*device, sobel.output);
        bind_ds(*device, triangulation.dset, sobel.output_sampler, sobel.output, 3);
}
        
void EditorRenderer::resize(const vk::Extent2D &new_extent)
{
        static vk::Format formats[] = {
                vk::Format::eR32G32B32A32Sfloat,
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
                formats[0], new_extent, tiling,
                usage, mem_flags, aspect
        };

        framebuffer_images.normal = ImageData {
                *phdev, *device,
                formats[1], new_extent, tiling,
                usage, mem_flags, aspect
        };

        framebuffer_images.material_index = ImageData {
                *phdev, *device,
                formats[3], new_extent, tiling,
                usage, mem_flags, aspect
        };

        depth_buffer = DepthBuffer {
                *phdev, *device,
                vk::Format::eD32Sfloat, new_extent
        };
       
        // Allocate Sobel filter output image
        sobel.output = ImageData {
                *phdev, *device,
                vk::Format::eR32Sfloat,
                vk::Extent2D { new_extent.width, new_extent.height},
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::ImageAspectFlagBits::eColor
        };

        // If needed, recreate framebuffer and rebind descriptor sets
        if (render_state.initialized) {
                std::vector <vk::ImageView> attachment_views {
                        *framebuffer_images.position.view,
                        *framebuffer_images.normal.view,
                        *framebuffer_images.material_index.view,
                        *depth_buffer.view
                };

                vk::FramebufferCreateInfo fb_info {
                        vk::FramebufferCreateFlags {},
                        *gbuffer.render_pass,
                        attachment_views,
                        new_extent.width, new_extent.height, 1
                };

                gbuffer_fb = vk::raii::Framebuffer {*device, fb_info};
        
                // Bind image to descriptor set
                std::array <vk::DescriptorImageInfo, 2> sobel_dset_image_infos {
                        vk::DescriptorImageInfo {
                                nullptr,
                                *framebuffer_images.material_index.view,
                                vk::ImageLayout::eGeneral
                        },

                        vk::DescriptorImageInfo {
                                nullptr,
                                *sobel.output.view,
                                vk::ImageLayout::eGeneral
                        },
                };

                std::array <vk::WriteDescriptorSet, 2> sobel_dset_writes {
                        vk::WriteDescriptorSet {
                                *sobel.dset,
                                0, 0,
                                vk::DescriptorType::eStorageImage,
                                sobel_dset_image_infos[0],
                        },

                        vk::WriteDescriptorSet {
                                *sobel.dset,
                                1, 0,
                                vk::DescriptorType::eStorageImage,
                                sobel_dset_image_infos[1],
                        },
                };

                device->updateDescriptorSets(sobel_dset_writes, nullptr);
        
                bind_ds(*device, triangulation.dset, framebuffer_images.position_sampler, framebuffer_images.position, 0);
                bind_ds(*device, triangulation.dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 1);
                bind_ds(*device, triangulation.dset, framebuffer_images.material_index_sampler, framebuffer_images.material_index, 2);
                bind_ds(*device, triangulation.dset, sobel.output_sampler, sobel.output, 3);

                bind_ds(*device, normal.dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 0);
        } else {
                // First time initialization

                // Create samplers for the framebuffer images
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
        }

        // Update extent
        extent = new_extent;
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

                        if (gbuffer.dset_refs.find(index) == gbuffer.dset_refs.end()) {
                                int new_index = gbuffer.dsets.size();
                                gbuffer.dset_refs[index] = new_index;
                                
                                gbuffer.dsets.emplace_back(
                                        std::move(
                                                vk::raii::DescriptorSets {
                                                        *device,
                                                        { **descriptor_pool, *gbuffer.dsl }
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

                                texture_loader->bind(gbuffer.dsets[new_index], albedo, 0);
                                texture_loader->bind(gbuffer.dsets[new_index], normal, 1);
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
                        *gbuffer.render_pass,
                        *gbuffer_fb,
                        vk::Rect2D { vk::Offset2D {}, extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *gbuffer.pipeline);

        GBuffer_PushConstants push_constants;

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
                        int dset_index = gbuffer.dset_refs[index];
                        render_info.cmd.bindDescriptorSets(
                                vk::PipelineBindPoint::eGraphics,
                                *gbuffer.pipeline_layout,
                                0, { *gbuffer.dsets[dset_index] }, {}
                        );

                        // Bind push constants
                        push_constants.model = transform.matrix();

                        int material_index = meshes[i].material_index;
                        push_constants.material_index = material_index;

                        Material material = Material::all[material_index];
                        
                        int texture_status = 0;
                        texture_status |= (material.has_albedo());
                        texture_status |= (material.has_normal() << 1);

                        push_constants.texture_status = texture_status;

                        render_info.cmd.pushConstants <GBuffer_PushConstants> (
                                *gbuffer.pipeline_layout,
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

        // Transition layout for position and normal map
        framebuffer_images.material_index.layout = vk::ImageLayout::ePresentSrcKHR;
        framebuffer_images.material_index.transition_layout(render_info.cmd, vk::ImageLayout::eGeneral);

        sobel.output.transition_layout(render_info.cmd, vk::ImageLayout::eGeneral);

        // Now apply Sobel filterto get edges
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *sobel.pipeline);
        render_info.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                *sobel.pipeline_layout,
                0, { *sobel.dset }, {}
        );

        render_info.cmd.dispatch(
                (extent.width + 15) / 16,
                (extent.height + 15) / 16,
                1
        );
}

void EditorRenderer::render_albedo(const RenderInfo &render_info, const std::vector <Entity> &entities)
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

                        if (albedo.dset_refs.find(index) == albedo.dset_refs.end()) {
                                int new_index = albedo.dsets.size();
                                albedo.dset_refs[index] = new_index;
                                
                                albedo.dsets.emplace_back(
                                        std::move(
                                                vk::raii::DescriptorSets {
                                                        *device,
                                                        { **descriptor_pool, *albedo.dsl }
                                                }.front()
                                        )
                                );

                                // Bind inforation to descriptor set
                                Material material = Material::all[meshes[i].material_index];
        
                                std::string albedo_src = "blank";
                                if (material.has_albedo())
                                        albedo_src = material.albedo_texture;

                                texture_loader->bind(albedo.dsets[new_index], albedo_src, 0);
                        }
                }
        }

        // Render to G-buffer
        // render_info.render_area.apply(render_info.cmd, render_info.extent);
        RenderArea::full().apply(render_info.cmd, render_info.extent);

        std::vector <vk::ClearValue> clear_values {
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } },
                vk::ClearDepthStencilValue { 1.0f, 0 }
        };

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *albedo.render_pass,
                        *render_info.framebuffer,
                        vk::Rect2D { vk::Offset2D {}, render_info.extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *albedo.pipeline);

        Albedo_PushConstants push_constants;

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
                        int dset_index = albedo.dset_refs[index];
                        render_info.cmd.bindDescriptorSets(
                                vk::PipelineBindPoint::eGraphics,
                                *albedo.pipeline_layout,
                                0, { *albedo.dsets[dset_index] }, {}
                        );

                        // Bind push constants
                        push_constants.model = transform.matrix();

                        int material_index = meshes[i].material_index;
                        Material material = Material::all[material_index];

                        push_constants.albedo = glm::vec4 { material.diffuse, 1.0f };
                        push_constants.has_albedo = material.has_albedo();

                        render_info.cmd.pushConstants <Albedo_PushConstants> (
                                *albedo.pipeline_layout,
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
}

void EditorRenderer::render_normals(const RenderInfo &render_info)
{
        // TODO: pass different render options

        // Render to screen
        render_info.render_area.apply(render_info.cmd, render_info.extent);

        std::vector <vk::ClearValue> clear_values {
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } },
                vk::ClearDepthStencilValue { 1.0f, 0 }
        };

        // TODO: remove depht buffer from this...

        // Transition framebuffer images to shader read
        // TODO: some way to avoid this hackery
        framebuffer_images.normal.layout = vk::ImageLayout::ePresentSrcKHR;
        framebuffer_images.normal.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *normal.render_pass,
                        *render_info.framebuffer,
                        vk::Rect2D { vk::Offset2D {}, render_info.extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *normal.pipeline);

        // Bind descriptor set
        render_info.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *normal.pipeline_layout,
                0, { *normal.dset }, {}
        );

        // Draw
        render_info.cmd.bindVertexBuffers(0, { *presentation_mesh.buffer }, { 0 });
        render_info.cmd.draw(6, 1, 0, 0);

        render_info.cmd.endRenderPass();
}

void EditorRenderer::render_triangulation(const RenderInfo &render_info)
{
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
        framebuffer_images.normal.layout = vk::ImageLayout::ePresentSrcKHR;
        framebuffer_images.material_index.layout = vk::ImageLayout::eGeneral;

        framebuffer_images.position.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        framebuffer_images.normal.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        framebuffer_images.material_index.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        sobel.output.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *triangulation.render_pass,
                        *render_info.framebuffer,
                        vk::Rect2D { vk::Offset2D {}, render_info.extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triangulation.pipeline);

        // Bind descriptor set
        render_info.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *triangulation.pipeline_layout,
                0, { *triangulation.dset }, {}
        );

        // Draw
        render_info.cmd.bindVertexBuffers(0, { *presentation_mesh.buffer }, { 0 });
        render_info.cmd.draw(6, 1, 0, 0);

        render_info.cmd.endRenderPass();
}

void EditorRenderer::render(const RenderInfo &render_info, const std::vector <Entity> &entities)
{
        // First render the G-Buffer
        // TODO: only if the mode requires it...
        render_gbuffer(render_info, entities);

        switch (render_state.mode) {
        case RenderState::eTriangulation:
                render_triangulation(render_info);
                break;

        // case RenderState::eWireframe:
        //         render_wireframe(render_info, entities);
        //         break;

        case RenderState::eNormals:
                render_normals(render_info);
                break;

        case RenderState::eAlbedo:
                render_albedo(render_info, entities);
                break;

        /* case RenderState::eSparseRTX:
                render_sparse_rtx(render_info, entities);
                break; */

        default:
                printf("ERROR: EditorRenderer::render called in invalid mode\n");
        }
}

void EditorRenderer::menu()
{
        static const char *modes[] = {
                "Triangulation",
                "Wireframe",
                "Normals",
                "Albedo",
                "SparseRTX"
        };

        if (ImGui::BeginMenu(modes[render_state.mode])) {
                if (ImGui::MenuItem("Triangulation"))
                        render_state.mode = RenderState::eTriangulation;

                // if (ImGui::MenuItem("Wireframe"))
                //         render_state.mode = RenderState::eWireframe;
                
                if (ImGui::MenuItem("Normals"))
                        render_state.mode = RenderState::eNormals;

                if (ImGui::MenuItem("Albedo"))
                        render_state.mode = RenderState::eAlbedo;

                // if (ImGui::MenuItem("SparseRTX"))
                //         render_state.mode = RenderState::eSparseRTX;

                ImGui::EndMenu();
        }
}
