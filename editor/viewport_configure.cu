// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>

// Engine headers
#include "editor_viewport.cuh"
#include "include/cuda/error.cuh"
#include "push_constants.hpp"

// Editor renderer shaders
extern const char *gbuffer_vert_shader;
extern const char *gbuffer_frag_shader;

// TODO: use a simple mesh shader for this...
extern const char *presentation_vert_shader;

extern const char *highlight_frag_shader;
extern const char *normal_frag_shader;
extern const char *uv_frag_shader;
extern const char *triangulation_frag_shader;

extern const char *albedo_vert_shader;
extern const char *albedo_frag_shader;

extern const char *bounding_box_vert_shader;
extern const char *bounding_box_frag_shader;

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

static const std::vector <DescriptorSetLayoutBinding> uv_bindings {
        // UV framebuffer
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

static const std::vector <DescriptorSetLayoutBinding> highlight_bindings {
        // Material index
	DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
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

void EditorViewport::configure_present()
{
        // Create a render pass for all presentation pipelines
        present_render_pass = make_render_pass(*device,
                { framebuffer_images.viewport.format },
                { vk::AttachmentLoadOp::eClear },
                vk::Format::eD32Sfloat,
                vk::AttachmentLoadOp::eClear
        );

        // Another framebuffer to render to the actual viewport
        std::vector <vk::ImageView> viewport_attachment_views {
                *framebuffer_images.viewport.view,
                *depth_buffer.view,
        };

        vk::FramebufferCreateInfo viewport_fb_info {
                vk::FramebufferCreateFlags {},
                *present_render_pass,
                viewport_attachment_views,
                extent.width, extent.height, 1
        };

        viewport_fb = vk::raii::Framebuffer {*device, viewport_fb_info};
}

// TODO: Rasterized and Raytraced backend for G-buffer generation...
void EditorViewport::configure_gbuffer_pipeline()
{
        // G-buffer render pass configuration
        std::vector <vk::Format> attachment_formats {
                framebuffer_images.position.format,
                framebuffer_images.normal.format,
                framebuffer_images.uv.format,
                framebuffer_images.material_index.format,
        };

        gbuffer_render_pass = make_render_pass(*device,
                attachment_formats,
                {
                        vk::AttachmentLoadOp::eClear,
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
                *framebuffer_images.uv.view,
                *framebuffer_images.material_index.view,
                *depth_buffer.view
        };

        vk::FramebufferCreateInfo fb_info {
                vk::FramebufferCreateFlags {},
                *gbuffer_render_pass,
                attachment_views,
                extent.width, extent.height, 1
        };

        gbuffer_fb = vk::raii::Framebuffer { *device, fb_info };

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
                *device, gbuffer_render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                Vertex::vertex_binding(),
                Vertex::vertex_attributes(),
                gbuffer.pipeline_layout,
        };

        gbuffer_grp_info.vertex_shader = std::move(*gbuffer_vertex.compile(*device));
        gbuffer_grp_info.fragment_shader = std::move(*gbuffer_fragment.compile(*device));
        gbuffer_grp_info.blend_attachments = { true, true, true, false };
        gbuffer_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        // Create the final pipeline
        gbuffer.pipeline = make_graphics_pipeline(gbuffer_grp_info);
}

void EditorViewport::configure_albedo_pipeline()
{
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
                *device, present_render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                Vertex::vertex_binding(),
                Vertex::vertex_attributes(),
                albedo.pipeline_layout,
        };

        albedo_grp_info.vertex_shader = std::move(*albedo_vertex.compile(*device));
        albedo_grp_info.fragment_shader = std::move(*albedo_fragment.compile(*device));
        albedo_grp_info.cull_mode = vk::CullModeFlagBits::eNone;
        
        // Create the final pipeline
        albedo.pipeline = make_graphics_pipeline(albedo_grp_info);
}

void EditorViewport::configure_normals_pipeline()
{
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
                *device, present_render_pass,
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

void EditorViewport::configure_uv_pipeline()
{
        uv.dsl = make_descriptor_set_layout(*device, uv_bindings);

        uv.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *uv.dsl, {}
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram uv_vertex {
                presentation_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram uv_fragment {
                uv_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo uv_grp_info {
                *device, present_render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                flat_vertex_binding,
                flat_vertex_attributes,
                uv.pipeline_layout,
        };

        uv_grp_info.vertex_shader = std::move(*uv_vertex.compile(*device));
        uv_grp_info.fragment_shader = std::move(*uv_fragment.compile(*device));
        uv_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        uv.pipeline = make_graphics_pipeline(uv_grp_info);
        
        // Configure descriptor set
        uv.dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *uv.dsl
                }
        }.front());
        
        bind_ds(*device, uv.dset, framebuffer_images.uv_sampler, framebuffer_images.uv, 0);
}

void EditorViewport::configure_triangulation_pipeline()
{
        // Triangulation pipeline
        triangulation.dsl = make_descriptor_set_layout(*device, triangulation_bindings);

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
                *device, present_render_pass,
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
        triangulation_grp_info.depth_test = false;
        triangulation_grp_info.depth_write = false;

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
        presentation_mesh_buffer = BufferData {
                *phdev, *device,
                vertices.size() * sizeof(FlatVertex),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible
                        | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        presentation_mesh_buffer.upload(vertices);

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

void EditorViewport::configure_bounding_box_pipeline()
{
        std::array <vk::PushConstantRange, 2> push_constant_ranges {
                vk::PushConstantRange {
                        vk::ShaderStageFlagBits::eVertex,
                        0, sizeof(BoundingBox_PushConstants)
                },

                vk::PushConstantRange {
                        vk::ShaderStageFlagBits::eFragment,
                        offsetof(BoundingBox_PushConstants, color),
                        sizeof(glm::vec4)
                }
        };

        bounding_box.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        nullptr,
                        push_constant_ranges
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram bounding_box_vertex {
                bounding_box_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram bounding_box_fragment {
                bounding_box_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };
        
        GraphicsPipelineInfo bounding_box_grp_info {
                *device, present_render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                Vertex::vertex_binding(), // TODO: refctor to vk_binding,
                // attributes...
                Vertex::vertex_attributes(),
                bounding_box.pipeline_layout
        };

        bounding_box_grp_info.vertex_shader = std::move(*bounding_box_vertex.compile(*device));
        bounding_box_grp_info.fragment_shader = std::move(*bounding_box_fragment.compile(*device));
        bounding_box_grp_info.cull_mode = vk::CullModeFlagBits::eNone;
        bounding_box_grp_info.polygon_mode = vk::PolygonMode::eLine;

        bounding_box.pipeline = make_graphics_pipeline(bounding_box_grp_info);

        // Create cube vertices, without index buffer
        Vertex v0 {{ -1.0f, -1.0f, -1.0f }};
        Vertex v1 {{  1.0f, -1.0f, -1.0f }};
        Vertex v2 {{  1.0f,  1.0f, -1.0f }};
        Vertex v3 {{ -1.0f,  1.0f, -1.0f }};
        Vertex v4 {{ -1.0f, -1.0f,  1.0f }};
        Vertex v5 {{  1.0f, -1.0f,  1.0f }};
        Vertex v6 {{  1.0f,  1.0f,  1.0f }};
        Vertex v7 {{ -1.0f,  1.0f,  1.0f }};

        std::vector <Vertex> cube_vertices {
                // Front face
                v0, v1, v2,
                v2, v3, v0,

                // Right face
                v1, v5, v6,
                v6, v2, v1,

                // Back face
                v7, v6, v5,
                v5, v4, v7,

                // Left face
                v4, v0, v3,
                v3, v7, v4,

                // Top face
                v4, v5, v1,
                v1, v0, v4,

                // Bottom face
                v3, v2, v6,
                v6, v7, v3
        };

        // Allocate the cube vertex buffer
        bounding_box.buffer = BufferData {
                *phdev, *device,
                cube_vertices.size() * sizeof(Vertex),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible
                        | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        bounding_box.buffer.upload(cube_vertices);
}

void EditorViewport::configure_sobel_pipeline()
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
        sobel.output_sampler = make_continuous_sampler(*device);
        bind_ds(*device, triangulation.dset, sobel.output_sampler, sobel.output, 3);
}

void EditorViewport::configure_highlight_pipeline()
{
        // Highlight pipeline
        highlight.dsl = make_descriptor_set_layout(*device, highlight_bindings);
        
        vk::PushConstantRange push_constant_range {
                vk::ShaderStageFlagBits::eFragment,
                0, sizeof(Highlight_PushConstants)
        };

        highlight.pipeline_layout = vk::raii::PipelineLayout {
                *device,
                vk::PipelineLayoutCreateInfo {
                        vk::PipelineLayoutCreateFlags {},
                        *highlight.dsl, push_constant_range
                }
        };

        // Load shaders and compile pipeline
        ShaderProgram highlight_vertex {
                presentation_vert_shader,
                vk::ShaderStageFlagBits::eVertex
        };

        ShaderProgram highlight_fragment {
                highlight_frag_shader,
                vk::ShaderStageFlagBits::eFragment
        };

        GraphicsPipelineInfo highlight_grp_info {
                *device, present_render_pass,
                nullptr, nullptr,
                nullptr, nullptr,
                flat_vertex_binding,
                flat_vertex_attributes,
                highlight.pipeline_layout
        };

        highlight_grp_info.vertex_shader = std::move(*highlight_vertex.compile(*device));
        highlight_grp_info.fragment_shader = std::move(*highlight_fragment.compile(*device));
        highlight_grp_info.cull_mode = vk::CullModeFlagBits::eNone;

        // Create pipeline
        highlight.pipeline = make_graphics_pipeline(highlight_grp_info);
        
        // Configure descriptor set
        highlight.dset = std::move(vk::raii::DescriptorSets {
                *device,
                vk::DescriptorSetAllocateInfo {
                        **descriptor_pool,
                        *highlight.dsl
                }
        }.front());

        bind_ds(*device, highlight.dset,
                framebuffer_images.material_index_sampler,
                framebuffer_images.material_index, 0);
}

// OptiX compilation options
static constexpr OptixPipelineCompileOptions pipeline_compile_options = {
	.usesMotionBlur = false,
	.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
	.numPayloadValues = 2,
	.numAttributeValues = 0,
	// .exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG,
	.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
	.pipelineLaunchParamsVariableName = "parameters",
	.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
};

static constexpr OptixModuleCompileOptions module_compile_options = {
	// .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
	// .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
	.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
	.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
};

static constexpr OptixPipelineLinkOptions pipeline_link_options = {
	.maxTraceDepth = 1,
};

// Configuring G-buffer raytracing pipeline
void EditorViewport::configure_gbuffer_rtx()
{
        static constexpr const char OPTIX_PTX_FILE[] = "bin/ptx/gbuffer_rtx_shader.ptx";

        // Create a context
        OptixDeviceContext context = system->context();

        std::string contents = common::read_file(OPTIX_PTX_FILE);
	
        char log[2048];
	size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(
                optixModuleCreate(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        contents.c_str(),
                        contents.size(),
                        log, &sizeof_log,
                        &gbuffer_rtx.module
                )
        );

        printf("Loaded module %p: %s\n", gbuffer_rtx.module, log);

        OptixProgramGroupOptions program_options = {};

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                desc.raygen.module = gbuffer_rtx.module;
                desc.raygen.entryFunctionName = "__raygen__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &gbuffer_rtx.ray_generation
                        )
                );

                printf("Loaded raygen %p: %s\n", gbuffer_rtx.ray_generation, log);
        }

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                desc.miss.module = gbuffer_rtx.module;
                desc.miss.entryFunctionName = "__miss__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &gbuffer_rtx.miss
                        )
                );

                printf("Loaded miss %p: %s\n", gbuffer_rtx.miss, log);
        }

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                desc.hitgroup.moduleCH = gbuffer_rtx.module;
                desc.hitgroup.entryFunctionNameCH = "__closesthit__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &gbuffer_rtx.closest_hit
                        )
                );

                printf("Loaded closest hit %p: %s\n", gbuffer_rtx.closest_hit, log);
        }

        // Create pipeline
        OptixProgramGroup program_groups[] = {
                gbuffer_rtx.ray_generation,
                gbuffer_rtx.closest_hit,
                gbuffer_rtx.miss,
        };

        OPTIX_CHECK(
                optixPipelineCreate(context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups, 3,
                        log, &sizeof_log,
                        &gbuffer_rtx.pipeline)
        );

        printf("Loaded pipeline %p: %s\n", gbuffer_rtx.pipeline, log);

        // Configure stacks
        OptixStackSizes stack_sizes = {};
        OPTIX_CHECK(optixUtilAccumulateStackSizes(gbuffer_rtx.closest_hit, &stack_sizes, gbuffer_rtx.pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(gbuffer_rtx.miss, &stack_sizes, gbuffer_rtx.pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(gbuffer_rtx.ray_generation, &stack_sizes, gbuffer_rtx.pipeline));

        uint32_t max_trace_depth = 1;
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 0;
        uint32_t direct_callable_stack_size_from_traversal = 0;
        uint32_t direct_callable_stack_size_from_state = 0;
        uint32_t continuation_stack_size = 0;

        OPTIX_CHECK(
                optixUtilComputeStackSizes(&stack_sizes,
                        max_trace_depth, max_cc_depth, max_dc_depth,
                        &direct_callable_stack_size_from_traversal,
                        &direct_callable_stack_size_from_state,
                        &continuation_stack_size)
        );

        OPTIX_CHECK(
                optixPipelineSetStackSize(gbuffer_rtx.pipeline,
                        direct_callable_stack_size_from_traversal,
                        direct_callable_stack_size_from_state,
                        continuation_stack_size, 2)
        );

        // Create shader binding table
        gbuffer_rtx.sbt = {};

        // Ray generation
        CUdeviceptr dev_raygen_record;

        optix::Record <void> raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(gbuffer_rtx.ray_generation, &raygen_record));

        CUDA_CHECK(cudaMalloc((void **) &dev_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_raygen_record, &raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        gbuffer_rtx.sbt.raygenRecord = dev_raygen_record;

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(gbuffer_rtx.miss, &miss_record));

        CUDA_CHECK(cudaMalloc((void **) &dev_miss_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_miss_record, &miss_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        gbuffer_rtx.sbt.missRecordBase = dev_miss_record;
        gbuffer_rtx.sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        gbuffer_rtx.sbt.missRecordCount = 1;

        gbuffer_rtx.sbt.hitgroupRecordBase = 0;
        gbuffer_rtx.sbt.hitgroupRecordStrideInBytes = 0;
        gbuffer_rtx.sbt.hitgroupRecordCount = 0;

        // Allocate parameters up front
        gbuffer_rtx.dev_launch_params = cuda::alloc(sizeof(GBufferParameters));

        gbuffer_rtx.launch_params = {};
        gbuffer_rtx.launch_params.io = optix_io_create();
}

// void EditorViewport::configure_amadeus_path_tracer(const Context &context)
// {
//         // TODO: adaptive resolution...
//         amadeus_path_tracer.armada = std::make_shared <amadeus::ArmadaRTX> (
//                 context, system, mesh_memory, vk::Extent2D { 1000, 1000 }
//         );
//
//         amadeus_path_tracer.armada->attach(
//                 "Path Tracer",
//                 std::make_shared <amadeus::PathTracer> ()
//                 // std::shared_ptr <amadeus::AttachmentRTX> (extra::load_attachment().ptr)
//         );
//
//         // Framer and related resources
// 	amadeus_path_tracer.framer = kobra::layers::Framer(context, present_render_pass);
//
// 	// Allocate necessary buffers
// 	size_t size = amadeus_path_tracer.armada->size();
//
// 	amadeus_path_tracer.dev_traced = kobra::cuda::alloc(size * sizeof(uint32_t));
// 	amadeus_path_tracer.traced.resize(size);
// }

void EditorViewport::configure_path_tracer(const Context &ctx)
{ 
        KOBRA_ASSERT(ctx.device != nullptr, "Editor (1): null device");

        static constexpr const char OPTIX_PTX_FILE[] = "bin/ptx/path_tracer.ptx";


        // Create a context
        OptixDeviceContext context = system->context();

        std::string contents = common::read_file(OPTIX_PTX_FILE);
	
        char log[2048];
	size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(
                optixModuleCreate(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        contents.c_str(),
                        contents.size(),
                        log, &sizeof_log,
                        &path_tracer.module
                )
        );

        KOBRA_ASSERT(ctx.device != nullptr, "Editor (1.1): null device");
        printf("Loaded module %p: %s\n", path_tracer.module, log);

        OptixProgramGroupOptions program_options = {};

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                desc.raygen.module = path_tracer.module;
                desc.raygen.entryFunctionName = "__raygen__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &path_tracer.ray_generation
                        )
                );

                printf("Loaded raygen %p: %s\n", path_tracer.ray_generation, log);
                KOBRA_ASSERT(ctx.device != nullptr, "Editor (1.12): null device");
        }

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                desc.miss.module = path_tracer.module;
                desc.miss.entryFunctionName = "__miss__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &path_tracer.miss
                        )
                );

                printf("Loaded miss %p: %s\n", path_tracer.miss, log);
                KOBRA_ASSERT(ctx.device != nullptr, "Editor (1.2): null device");
        }

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                desc.hitgroup.moduleCH = path_tracer.module;
                desc.hitgroup.entryFunctionNameCH = "__closesthit__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &path_tracer.closest_hit
                        )
                );

                printf("Loaded closest hit %p: %s\n", path_tracer.closest_hit, log);
                KOBRA_ASSERT(ctx.device != nullptr, "Editor (1.3): null device");
        }

        // Create pipeline
        OptixProgramGroup program_groups[] = {
                path_tracer.ray_generation,
                path_tracer.closest_hit,
                path_tracer.miss,
        };

        OPTIX_CHECK(
                optixPipelineCreate(context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups, 3,
                        log, &sizeof_log,
                        &path_tracer.pipeline)
        );

        printf("Loaded pipeline %p: %s\n", path_tracer.pipeline, log);
        KOBRA_ASSERT(ctx.device != nullptr, "Editor (2): null device");

        // Configure stacks
        OptixStackSizes stack_sizes = {};
        OPTIX_CHECK(optixUtilAccumulateStackSizes(path_tracer.closest_hit, &stack_sizes, path_tracer.pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(path_tracer.miss, &stack_sizes, path_tracer.pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(path_tracer.ray_generation, &stack_sizes, path_tracer.pipeline));

        uint32_t max_trace_depth = 1;
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 0;
        uint32_t direct_callable_stack_size_from_traversal = 0;
        uint32_t direct_callable_stack_size_from_state = 0;
        uint32_t continuation_stack_size = 0;

        OPTIX_CHECK(
                optixUtilComputeStackSizes(&stack_sizes,
                        max_trace_depth, max_cc_depth, max_dc_depth,
                        &direct_callable_stack_size_from_traversal,
                        &direct_callable_stack_size_from_state,
                        &continuation_stack_size)
        );

        OPTIX_CHECK(
                optixPipelineSetStackSize(path_tracer.pipeline,
                        direct_callable_stack_size_from_traversal,
                        direct_callable_stack_size_from_state,
                        continuation_stack_size, 2)
        );
        KOBRA_ASSERT(ctx.device != nullptr, "Editor (3): null device");

        // Create shader binding table
        path_tracer.sbt = {};

        // Ray generation
        CUdeviceptr dev_raygen_record;

        optix::Record <void> raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(path_tracer.ray_generation, &raygen_record));

        CUDA_CHECK(cudaMalloc((void **) &dev_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_raygen_record, &raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        path_tracer.sbt.raygenRecord = dev_raygen_record;

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(path_tracer.miss, &miss_record));

        CUDA_CHECK(cudaMalloc((void **) &dev_miss_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_miss_record, &miss_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        path_tracer.sbt.missRecordBase = dev_miss_record;
        path_tracer.sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        path_tracer.sbt.missRecordCount = 1;

        path_tracer.sbt.hitgroupRecordBase = 0;
        path_tracer.sbt.hitgroupRecordStrideInBytes = 0;
        path_tracer.sbt.hitgroupRecordCount = 0;

        // Allocate parameters up front
        path_tracer.dev_launch_params = cuda::alloc(sizeof(PathTracerParameters));

        // path_tracer.launch_params = {};
        path_tracer.launch_params.io = optix_io_create();
	
       
        // TODO: render from raw float4 data...
        KOBRA_ASSERT(ctx.device != nullptr, "Editor (-1): null device");
}

void initialize(SparseGI *sparse_gi, const Context &ctx, const OptixDeviceContext &context)
{
        static constexpr const char OPTIX_PTX_FILE[] = "bin/ptx/sparse_gi.ptx";

        std::string contents = common::read_file(OPTIX_PTX_FILE);
	
        char log[2048];
	size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(
                optixModuleCreate(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        contents.c_str(),
                        contents.size(),
                        log, &sizeof_log,
                        &sparse_gi->module
                )
        );

        printf("Loaded module %p: %s\n", sparse_gi->module, log);

        OptixProgramGroupOptions program_options = {};

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                desc.raygen.module = sparse_gi->module;
                desc.raygen.entryFunctionName = "__raygen__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &sparse_gi->ray_generation
                        )
                );

                printf("Loaded raygen %p: %s\n", sparse_gi->ray_generation, log);
        }

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                desc.miss.module = sparse_gi->module;
                desc.miss.entryFunctionName = "__miss__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &sparse_gi->miss
                        )
                );

                printf("Loaded miss %p: %s\n", sparse_gi->miss, log);
        }

        {
                OptixProgramGroupDesc desc = {};
                desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                desc.hitgroup.moduleCH = sparse_gi->module;
                desc.hitgroup.entryFunctionNameCH = "__closesthit__";

                OPTIX_CHECK(
                        optixProgramGroupCreate(
                                context,
                                &desc, 1,
                                &program_options,
                                log, &sizeof_log,
                                &sparse_gi->closest_hit
                        )
                );

                printf("Loaded closest hit %p: %s\n", sparse_gi->closest_hit, log);
        }

        // Create pipeline
        OptixProgramGroup program_groups[] = {
                sparse_gi->ray_generation,
                sparse_gi->closest_hit,
                sparse_gi->miss,
        };

        OPTIX_CHECK(
                optixPipelineCreate(context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups, 3,
                        log, &sizeof_log,
                        &sparse_gi->pipeline)
        );

        printf("Loaded pipeline %p: %s\n", sparse_gi->pipeline, log);

        // Configure stacks
        OptixStackSizes stack_sizes = {};
        OPTIX_CHECK(optixUtilAccumulateStackSizes(sparse_gi->closest_hit, &stack_sizes, sparse_gi->pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(sparse_gi->miss, &stack_sizes, sparse_gi->pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(sparse_gi->ray_generation, &stack_sizes, sparse_gi->pipeline));

        uint32_t max_trace_depth = 1;
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 0;
        uint32_t direct_callable_stack_size_from_traversal = 0;
        uint32_t direct_callable_stack_size_from_state = 0;
        uint32_t continuation_stack_size = 0;

        OPTIX_CHECK(
                optixUtilComputeStackSizes(&stack_sizes,
                        max_trace_depth, max_cc_depth, max_dc_depth,
                        &direct_callable_stack_size_from_traversal,
                        &direct_callable_stack_size_from_state,
                        &continuation_stack_size)
        );

        OPTIX_CHECK(
                optixPipelineSetStackSize(sparse_gi->pipeline,
                        direct_callable_stack_size_from_traversal,
                        direct_callable_stack_size_from_state,
                        continuation_stack_size, 2)
        );

        // Create shader binding table
        sparse_gi->sbt = {};

        // Ray generation
        CUdeviceptr dev_raygen_record;

        optix::Record <void> raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(sparse_gi->ray_generation, &raygen_record));

        CUDA_CHECK(cudaMalloc((void **) &dev_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_raygen_record, &raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        sparse_gi->sbt.raygenRecord = dev_raygen_record;

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(sparse_gi->miss, &miss_record));

        CUDA_CHECK(cudaMalloc((void **) &dev_miss_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_miss_record, &miss_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        sparse_gi->sbt.missRecordBase = dev_miss_record;
        sparse_gi->sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        sparse_gi->sbt.missRecordCount = 1;

        sparse_gi->sbt.hitgroupRecordBase = 0;
        sparse_gi->sbt.hitgroupRecordStrideInBytes = 0;
        sparse_gi->sbt.hitgroupRecordCount = 0;

        // Allocate parameters up front
        sparse_gi->launch_params = {};
        sparse_gi->dev_launch_params = cuda::alloc(sizeof(SparseGIParameters));
}
