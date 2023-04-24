#include "common.hpp"
#include "include/cuda/cast.cuh"
#include "include/cuda/error.cuh"
#include "push_constants.hpp"

__global__ void overwrite_normals(cudaSurfaceObject_t normal, vk::Extent2D extent, OptixIO io)
{
        int x0 = blockDim.x * blockIdx.x + threadIdx.x;
        int y0 = blockDim.y * blockIdx.y + threadIdx.y;

        for (int x = x0; x < extent.width; x += blockDim.x * gridDim.x) {
                for (int y = y0; y < extent.height; y += blockDim.y * gridDim.y) {
                        if (x < extent.width && y < extent.height) {
                                if (x == 0 && y == 0) {
                                        optix_io_write_str(&io, "(x, y) matches (0, 0)\n");
                                }

                                float4 color { 0.0f, 0.0f, 0.0f, 0.0f };
                                color.x = (float) x / (float) extent.width;
                                color.y = (float) y / (float) extent.height;
                                surf2Dwrite(color, normal, x * sizeof(float4), y);
                        }
                }
        }
}

void EditorViewport::render_gbuffer(const RenderInfo &render_info, const std::vector <Entity> &entities)
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

        /* Render to G-buffer
        RenderArea::full().apply(render_info.cmd, extent);

        std::vector <vk::ClearValue> clear_values {
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 0.0f } },
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 0.0f } },
                vk::ClearColorValue { std::array <int32_t, 4> { -1, -1, -1, -1 }},
                vk::ClearDepthStencilValue { 1.0f, 0 }
        };

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *gbuffer_render_pass,
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

        render_info.cmd.endRenderPass(); */
        
        // Load SBT information
        bool rebuild_tlas = false;
        bool rebuild_sbt = false;

        for (const auto &entity : entities) {
                int id = entity.id;
                const auto &renderable = entity.get <Renderable> ();
                const auto &transform = entity.get <Transform> ();

                // Generate cache data
                mesh_memory->cache_cuda(entity);

                // Create SBT record for each submesh
                const auto &meshes = renderable.mesh->submeshes;
                for (int i = 0; i < meshes.size(); i++) {
                        MeshIndex index { id, i };

                        if (gbuffer_rtx.record_refs.find(index) == gbuffer_rtx.record_refs.end()) {
                                int new_index = gbuffer_rtx.linearized_records.size();
                                gbuffer_rtx.record_refs[index] = new_index;
                                gbuffer_rtx.linearized_records.emplace_back();

                                optix::Record <Hit> record;
                                optix::pack_header(gbuffer_rtx.closest_hit, record);

                                auto cachelet = mesh_memory->get(entity.id, i);

                                record.data.vertices = cachelet.m_cuda_vertices;
                                record.data.triangles = (uint3 *) cachelet.m_cuda_triangles;
                                record.data.model = transform.matrix();
                                record.data.index = meshes[i].material_index;

                                gbuffer_rtx.linearized_records[new_index] = record;

                                rebuild_tlas = true;
                                rebuild_sbt = true;
                        }

                        // TODO: checking for transformations...
                }
        }

        // Trigger update for System
        system->update(entities);

        if (rebuild_tlas) {
                gbuffer_rtx.launch_params.handle
                        = system->build_tlas(1, gbuffer_rtx.record_refs);
        }

        if (rebuild_sbt) {
                printf("Rebuilding SBT\n");
                if (gbuffer_rtx.sbt.hitgroupRecordBase)
                        cuda::free(gbuffer_rtx.sbt.hitgroupRecordBase);
        
                gbuffer_rtx.sbt.hitgroupRecordBase = cuda::make_buffer_ptr(gbuffer_rtx.linearized_records);
                gbuffer_rtx.sbt.hitgroupRecordStrideInBytes = sizeof(optix::Record <Hit>);
                gbuffer_rtx.sbt.hitgroupRecordCount = gbuffer_rtx.linearized_records.size();
        }

        // Configure launch parameters
        auto uvw = uvw_frame(render_info.camera, render_info.camera_transform);

        gbuffer_rtx.launch_params.U = cuda::to_f3(uvw.u);
        gbuffer_rtx.launch_params.V = cuda::to_f3(uvw.v);
        gbuffer_rtx.launch_params.W = cuda::to_f3(uvw.w);
        
        gbuffer_rtx.launch_params.origin = cuda::to_f3(render_info.camera_transform.position);
        gbuffer_rtx.launch_params.resolution = { extent.width, extent.height };
        
        gbuffer_rtx.launch_params.position_surface = framebuffer_images.cu_position_surface;
        gbuffer_rtx.launch_params.normal_surface = framebuffer_images.cu_normal_surface;
        gbuffer_rtx.launch_params.index_surface = framebuffer_images.cu_material_index_surface;

        // cuda::copy(gbuffer_rtx.dev_launch_params, &gbuffer_rtx.launch_params, 1);
        GBufferParameters *dev_params = (GBufferParameters *) gbuffer_rtx.dev_launch_params;
        cudaMemcpy(dev_params, &gbuffer_rtx.launch_params, sizeof(GBufferParameters), cudaMemcpyHostToDevice);
        
        // Launch G-buffer raytracing pipeline
        // printf("Pipeline: %p\n", gbuffer_rtx.pipeline);
        
        OPTIX_CHECK(
                optixLaunch(
                        gbuffer_rtx.pipeline, 0,
                        gbuffer_rtx.dev_launch_params,
                        sizeof(GBufferParameters),
                        &gbuffer_rtx.sbt,
                        extent.width, extent.height, 1
                )
        );

        CUDA_SYNC_CHECK();

        std::string output = optix_io_read(&gbuffer_rtx.launch_params.io);
        std::cout << "Str size: " << output.size() << "\n";
        std::cout << "Output: \"" << output << "\"\n";
        printf("Finished launching G-buffer RTX pipeline\n");

        /* Overwrite normals
        dim3 block_size { 16, 16 };
        dim3 grid_size {
                (extent.width + block_size.x - 1) / block_size.x,
                (extent.height + block_size.y - 1) / block_size.y
        };

        overwrite_normals <<< grid_size, block_size >>> (
                framebuffer_images.cu_normal_surface,
                extent, gbuffer_rtx.launch_params.io
        );

        CUDA_SYNC_CHECK();
        
        output = optix_io_read(&gbuffer_rtx.launch_params.io);
        // std::cout << "Again: Str size: " << output.size() << "\n";
        // std::cout << "Output: \"" << output << "\"\n"; */

        optix_io_clear(&gbuffer_rtx.launch_params.io);

        // Transition layout for position and normal map
        // framebuffer_images.material_index.layout = vk::ImageLayout::ePresentSrcKHR;
        // framebuffer_images.material_index.transition_layout(render_info.cmd, vk::ImageLayout::eGeneral);
        framebuffer_images.material_index.layout = vk::ImageLayout::eUndefined;
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

        // Transition framebuffer images to shader read for later stages
        // TODO: some way to avoid this hackery
        // framebuffer_images.position.layout = vk::ImageLayout::ePresentSrcKHR;
        // framebuffer_images.normal.layout = vk::ImageLayout::ePresentSrcKHR;
        framebuffer_images.position.layout = vk::ImageLayout::eUndefined;
        framebuffer_images.normal.layout = vk::ImageLayout::eUndefined;
        framebuffer_images.material_index.layout = vk::ImageLayout::eGeneral;

        framebuffer_images.position.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        framebuffer_images.normal.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        framebuffer_images.material_index.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        sobel.output.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void EditorViewport::render_path_traced
                        (const RenderInfo &render_info,
                        const std::vector <Entity> &entities,
                        daemons::Transform &transform_daemon)
{
        // Run the path tracer, then extract image and blit to viewport
        path_tracer.armada->render(
                entities,
                transform_daemon,
                render_info.camera,
                render_info.camera_transform,
                false
        );
			
        float4 *buffer = (float4 *) path_tracer.armada->color_buffer();

        vk::Extent2D rtx_extent = path_tracer.armada->extent();
        kobra::cuda::hdr_to_ldr(
                buffer,
                (uint32_t *) path_tracer.dev_traced,
                rtx_extent.width, rtx_extent.height,
                kobra::cuda::eTonemappingACES
        );

        kobra::cuda::copy(
                path_tracer.traced, path_tracer.dev_traced,
                path_tracer.armada->size() * sizeof(uint32_t)
        );

        path_tracer.framer.pre_render(
                render_info.cmd,
                kobra::RawImage {
                        .data = path_tracer.traced,
                        .width = rtx_extent.width,
                        .height = rtx_extent.height,
                        .channels = 4
                }
        );

        // Start render pass and blit
        // TODO: function to start render pass for presentation
        render_info.render_area.apply(render_info.cmd, render_info.extent);

        std::vector <vk::ClearValue> clear_values {
                vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } },
                vk::ClearDepthStencilValue { 1.0f, 0 }
        };

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *present_render_pass,
                        *viewport_fb,
                        vk::Rect2D { vk::Offset2D {}, render_info.extent },
                        clear_values
                },
                vk::SubpassContents::eInline
        );

        // TODO: import CUDA to Vulkan and render straight to the image
        path_tracer.framer.render(render_info.cmd);
}

void EditorViewport::render_albedo(const RenderInfo &render_info, const std::vector <Entity> &entities)
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
                        *present_render_pass,
                        *viewport_fb,
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

        // render_info.cmd.endRenderPass();
}

void EditorViewport::render_normals(const RenderInfo &render_info)
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
        // framebuffer_images.normal.layout = vk::ImageLayout::ePresentSrcKHR;
        // framebuffer_images.normal.transition_layout(render_info.cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *present_render_pass,
                        *viewport_fb,
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
        render_info.cmd.bindVertexBuffers(0, { *presentation_mesh_buffer.buffer }, { 0 });
        render_info.cmd.draw(6, 1, 0, 0);

        // render_info.cmd.endRenderPass();
}

void EditorViewport::render_triangulation(const RenderInfo &render_info)
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

        render_info.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *present_render_pass,
                        *viewport_fb,
                        vk::Rect2D { vk::Offset2D {}, render_info.extent },
                        // {}
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
        render_info.cmd.bindVertexBuffers(0, { *presentation_mesh_buffer.buffer }, { 0 });
        render_info.cmd.draw(6, 1, 0, 0);
}

void EditorViewport::render_bounding_box(const RenderInfo &render_info, const std::vector <Entity> &entities)
{
        // NOTE: Showing bounding boxes does not start a new render pass

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *bounding_box.pipeline);

        // Push constants
        BoundingBox_PushConstants push_constants;
        
        push_constants.view = render_info.camera.view_matrix(render_info.camera_transform);
        push_constants.projection = render_info.camera.perspective_matrix();
        push_constants.color = glm::vec4 { 0.0f, 1.0f, 0.0f, 1.0f };

        // Draw
        // TODO: instancing...
        for (auto &entity : entities) {
                const auto &transform = entity.get <Transform> ();
                const auto &renderable = entity.get <Renderable> ();
                const auto &meshes = renderable.mesh->submeshes;

                for (int i = 0; i < meshes.size(); i++) {
                        // Bind push constants

                        // TODO: cache the bounding box...
                        BoundingBox bbox;

                        auto index = std::make_pair(entity.id, i);
                        if (bounding_box.cache.find(index) != bounding_box.cache.end()) {
                                bbox = bounding_box.cache[index];
                        } else {
                                bbox = meshes[i].bbox();
                                bounding_box.cache[index] = bbox;
                        }

                        bbox = bbox.transform(transform);

                        Transform mesh_transform;
                        mesh_transform.position = (bbox.min + bbox.max)/2.0f;
                        mesh_transform.scale = (bbox.max - bbox.min)/2.0f;

                        // TODO: bbox per entity or submesh?
                        push_constants.model = mesh_transform.matrix();

                        render_info.cmd.pushConstants <BoundingBox_PushConstants> (
                                *bounding_box.pipeline_layout,
                                vk::ShaderStageFlagBits::eVertex,
                                0, push_constants
                        );

                        // Draw
                        render_info.cmd.bindVertexBuffers(0, { *bounding_box.buffer.buffer }, { 0 });
                        render_info.cmd.draw(bounding_box.buffer.size/sizeof(Vertex), 1, 0, 0);
                }
        }
}

void EditorViewport::render_highlight(const RenderInfo &render_info, const std::vector <Entity> &entities)
{
        // If none are highlighted, return
        if (render_info.highlighted_entities.empty()) {
                render_info.cmd.endRenderPass();
                return;
        }

        // Push constants
        Highlight_PushConstants push_constants;

        push_constants.color = { 0.89, 0.73, 0.33, 0.5 };
        if (render_state.mode == RenderState::eTriangulation)
                push_constants.color = { 0.0, 0.0, 0.0, 0.3 };
        else if (render_state.mode == RenderState::eNormals)
                push_constants.color = { 0.0, 0.0, 0.0, 0.3 };

        push_constants.material_index = -1;
        push_constants.material_index = *render_info.highlighted_entities.begin();

        // Start the pipeline
        render_info.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *highlight.pipeline);
        
        // Bind descriptor set
        render_info.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *highlight.pipeline_layout,
                0, { *highlight.dset }, {}
        );

        // Bind push constants
        render_info.cmd.pushConstants <Highlight_PushConstants> (
                *highlight.pipeline_layout,
                vk::ShaderStageFlagBits::eFragment,
                0, push_constants
        );

        // Draw
        render_info.cmd.bindVertexBuffers(0, { *presentation_mesh_buffer.buffer }, { 0 });
        render_info.cmd.draw(6, 1, 0, 0);

        render_info.cmd.endRenderPass();
}

void EditorViewport::render(const RenderInfo &render_info, const std::vector <Entity> &entities, daemons::Transform &transform_daemon)
{
        switch (render_state.mode) {
        case RenderState::eTriangulation:
                render_gbuffer(render_info, entities);
                render_triangulation(render_info);
                break;

        case RenderState::eNormals:
                render_gbuffer(render_info, entities);
                render_normals(render_info);
                break;

        case RenderState::eAlbedo:
                render_albedo(render_info, entities);
                break;

        case RenderState::ePathTraced:
                render_path_traced(render_info, entities, transform_daemon);
                break;

        default:
                printf("ERROR: EditorViewport::render called in invalid mode\n");
        }

        // If bounding boxes are enabled, render them
        if (render_state.bounding_boxes)
                render_bounding_box(render_info, entities);

        // Render the highlight
        render_highlight(render_info, entities);
}

std::vector <std::pair <int, int>>
EditorViewport::selection_query(const std::vector <Entity> &entities, const glm::vec2 &loc)
{
        // Copy the material index image
        vk::raii::Queue queue {*device, 0, 0};

        submit_now(*device, queue, *command_pool,
                [&](const vk::raii::CommandBuffer &cmd) {
                        transition_image_layout(cmd,
                                *framebuffer_images.material_index.image,
                                framebuffer_images.material_index.format,
                                vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::ImageLayout::eTransferSrcOptimal
                        );

                        copy_image_to_buffer(cmd,
                                framebuffer_images.material_index.image,
                                index_staging_buffer.buffer,
                                framebuffer_images.material_index.format,
                                extent.width, extent.height
                        );

                        transition_image_layout(cmd,
                                *framebuffer_images.material_index.image,
                                framebuffer_images.material_index.format,
                                vk::ImageLayout::eTransferSrcOptimal,
                                vk::ImageLayout::eShaderReadOnlyOptimal
                        );
                }
        );

        // Download to host
        index_staging_data.resize(index_staging_buffer.size/sizeof(uint32_t));
        index_staging_buffer.download(index_staging_data);

        // Find the ID of the entity at the location
        glm::ivec2 iloc = loc * glm::vec2(extent.width, extent.height);
        uint32_t id = index_staging_data[iloc.y * extent.width + iloc.x];
        uint32_t material_id = id & 0xFFFF; // Lower 16 bits

        // Find the index of the entity in the selection
        std::vector <std::pair <int, int>> entity_indices;
        for (auto &entity : entities) {
                const Renderable &renderable = entity.get <Renderable> ();

                auto &meshes = renderable.mesh->submeshes;
                for (int j = 0; j < meshes.size(); j++) {
                        if (meshes[j].material_index == material_id) {
                                entity_indices.push_back({ entity.id, j });
                                break;
                        }
                }
        }

        return entity_indices;
}

void EditorViewport::menu()
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

                if (ImGui::MenuItem("Path Traced"))
                        render_state.mode = RenderState::ePathTraced;

                if (ImGui::Checkbox("Show bounding boxes", &render_state.bounding_boxes));

                ImGui::EndMenu();
        }
}
