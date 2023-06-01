#include "editor_viewport.cuh"
#include "include/cuda/cast.cuh"
#include "include/cuda/error.cuh"
#include "include/daemons/material.hpp"
#include "optix/sparse_gi_shader.cuh"

// Gaussian blur of irradiance per pixel
__global__
void gauss_filter(float4 *irradiance, float4 *out, vk::Extent2D extent)
{
        constexpr int N = 5;
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = -N; i <= N; i++) {
                for (int j = -N; j <= N; j++) {
                        int2 offset = make_int2(i, j);
                        int2 pos = make_int2(x % extent.width, x / extent.width);
                        int2 new_pos = pos + offset;

                        if (new_pos.x < 0 || new_pos.x >= extent.width ||
                                new_pos.y < 0 || new_pos.y >= extent.height)
                                continue;

                        int new_index = new_pos.x + new_pos.y * extent.width;
                        sum += make_float3(irradiance[new_index]);
                }
        }

        out[x] = make_float4(sum / ((2 * N + 1) * (2 * N + 1)), 0.0f);

        // TODO: better filter that accounts for normals
}

__device__
void convert_material(const cuda::_material &src, cuda::Material &dst, float2 uv)
{
        dst.diffuse = src.diffuse;
        dst.specular = src.specular;
        dst.emission = src.emission;
        dst.roughness = src.roughness;
        dst.refraction = src.refraction;
        dst.type = src.type;

        if (src.textures.has_diffuse) {
                float4 diffuse = tex2D <float4> (src.textures.diffuse, uv.x, uv.y);
                dst.diffuse = make_float3(diffuse);
        }
}

struct FinalGatherInfo {
        cuda::_material *materials;
        float time;
        float3 *averaged_directions;
        float3 camera;
        float4 *color;

        Reservoir <DirectLightingSample> *direct_lighting;
        float4 *indirect_irradiance;
                
        cudaSurfaceObject_t position_surface;
        cudaSurfaceObject_t normal_surface;
        cudaSurfaceObject_t uv_surface;
        cudaSurfaceObject_t index_surface;
        
        vk::Extent2D extent;

        bool direct;
        bool indirect;
};

__global__
void final_gather(FinalGatherInfo info)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        int x = index % info.extent.width;
        int y = index / info.extent.width;

        if (x >= info.extent.width || y >= info.extent.height)
                return;

        // Retrieve surface data
        float4 raw_position;
        float4 raw_normal;
        float4 raw_uv;
        int32_t raw_index;

        surf2Dread(&raw_position, info.position_surface, x * sizeof(float4),
                   info.extent.height - (y + 1));
        surf2Dread(&raw_normal, info.normal_surface, x * sizeof(float4), info.extent.height - (y + 1));
        surf2Dread(&raw_uv, info.uv_surface, x * sizeof(float4), info.extent.height - (y + 1));
        surf2Dread(&raw_index, info.index_surface, x * sizeof(int32_t), info.extent.height - (y + 1));

        // If there is a miss, then exit...
        if (raw_index == -1) {
                // TODO: skybox
                info.color[index] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
                return;
        }

        int32_t triangle_id = raw_index >> 16;
        int32_t material_id = raw_index & 0xFFFF;

        // Reconstruct the surface hit
        float3 position = { raw_position.x, raw_position.y, raw_position.z };
        float3 normal = { raw_normal.x, raw_normal.y, raw_normal.z };
        float2 uv = { raw_uv.x, 1 - raw_uv.y };

        // Correct the normal
        float3 ray = position - info.camera;
        if (dot(ray, normal) > 0.0f)
                normal = -normal;

        float3 seed = make_float3(x, y, info.time);

        cuda::_material m = info.materials[material_id];

        cuda::SurfaceHit sh;
        sh.x = position;
        sh.wo = normalize(info.camera - position);
        sh.n = normalize(normal);
        sh.entering = (raw_normal.w > 0.0f);

        convert_material(m, sh.mat, uv);
        float sign = (sh.mat.type == eTransmission) ? -1.0f : 1.0f;
        sh.x += sign * sh.n * 1e-3f;

        // Get brdf value
        // TODO: get weighted average of directions (based on pdfs)
        float3 wi = info.averaged_directions[index];
        float3 brdf = cuda::brdf(sh, wi, eDiffuse);
        float pdf = cuda::pdf(sh, wi, eDiffuse);

        float3 indirect = make_float3(info.indirect_irradiance[index]);
        float3 direct = info.direct_lighting[index].data.Le;
        float3 color = info.direct * direct + info.indirect * indirect;
        info.color[index] = make_float4(color, 1.0f);
}

void SparseGI::render(EditorViewport *ev,
                const RenderInfo &render_info,
                const std::vector <Entity> &entities,
                const MaterialDaemon *md)
{
        const Camera &camera = render_info.camera;
        const Transform &camera_transform = render_info.camera_transform;
        
        // Handle resizing
        if (resize_queue.size() > 0) {
                vk::Extent2D new_extent = resize_queue.back();
                resize_queue = {};

                if (launch_params.previous_position != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.previous_position));

                if (launch_params.indirect.screen_irradiance != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.indirect.screen_irradiance));

                if (launch_params.indirect.irradiance_directions != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.indirect.irradiance_directions));

                if (launch_params.indirect.final_irradiance != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.indirect.final_irradiance));

                if (launch_params.direct_lighting != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.direct_lighting));

                int size = new_extent.width * new_extent.height;
                CUDA_CHECK(cudaMalloc((void **) &launch_params.previous_position, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.screen_irradiance, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.final_irradiance, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.irradiance_directions, size * sizeof(float3)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.direct_lighting, size * sizeof(Reservoir <DirectLightingSample>)));
        }

        // Configure launch parameters
        // launch_params.color = ev->common_rtx.dev_color;
        launch_params.time = ev->common_rtx.timer.elapsed_start();
        launch_params.dirty = render_info.camera_transform_dirty;
        launch_params.reset = ev->render_state.sparse_gi_reset
                        | ev->common_rtx.material_reset
                        | manual_reset;
        launch_params.samples++;
                
        if (launch_params.reset)
                manual_reset = false;

        ev->render_state.sparse_gi_reset = false;
        if (ev->render_state.sparse_gi_reset) {
                launch_params.previous_view = camera.view_matrix(camera_transform);
                launch_params.previous_projection = camera.perspective_matrix();
                launch_params.samples = 0;
        }

        auto uvw = uvw_frame(camera, camera_transform);

        launch_params.U = cuda::to_f3(uvw.u);
        launch_params.V = cuda::to_f3(uvw.v);
        launch_params.W = cuda::to_f3(uvw.w);
        
        launch_params.origin = cuda::to_f3(render_info.camera_transform.position);
        launch_params.resolution = { ev->extent.width, ev->extent.height };
        
        launch_params.position_surface = ev->framebuffer_images.cu_position_surface;
        launch_params.normal_surface = ev->framebuffer_images.cu_normal_surface;
        launch_params.uv_surface = ev->framebuffer_images.cu_uv_surface;
        launch_params.index_surface = ev->framebuffer_images.cu_material_index_surface;

        launch_params.materials = (cuda::_material *) ev->common_rtx.dev_materials;

        // TODO: move to final gather
        launch_params.sky.texture = ev->environment_map.texture;
        launch_params.sky.enabled = ev->environment_map.valid;
        
        SparseGIParameters *dev_params = (SparseGIParameters *) dev_launch_params;
        CUDA_CHECK(cudaMemcpy(dev_params, &launch_params, sizeof(SparseGIParameters), cudaMemcpyHostToDevice));
        
        OPTIX_CHECK(
                optixLaunch(
                        pipeline, 0,
                        dev_launch_params,
                        sizeof(SparseGIParameters),
                        &sbt,
                        ev->extent.width, ev->extent.height, 1
                )
        );

        // TODO: afterward perform a brdf convolution
        // TODO: separate visibility for direct lighting?

        CUDA_SYNC_CHECK();

        // Final gather
        if (false && launch_params.samples > 0) {
                gauss_filter <<< (ev->extent.width * ev->extent.height + 255) / 256, 256 >>> (
                        launch_params.indirect.screen_irradiance,
                        launch_params.indirect.final_irradiance,
                        ev->extent
                );
        } else {
                CUDA_CHECK(cudaMemcpy(
                        launch_params.indirect.final_irradiance,
                        launch_params.indirect.screen_irradiance,
                        ev->extent.width * ev->extent.height * sizeof(float4),
                        cudaMemcpyDeviceToDevice
                ));
        }

        FinalGatherInfo info;
        info.averaged_directions = launch_params.indirect.irradiance_directions;
        info.camera = launch_params.origin;
        info.color = ev->common_rtx.dev_color;
        info.direct_lighting = launch_params.direct_lighting;
        info.extent = ev->extent;
        info.index_surface = launch_params.index_surface;
        info.indirect_irradiance = launch_params.indirect.final_irradiance;
        info.materials = (cuda::_material *) ev->common_rtx.dev_materials;
        info.normal_surface = launch_params.normal_surface;
        info.position_surface = launch_params.position_surface;
        info.time = launch_params.time;
        info.uv_surface = launch_params.uv_surface;
        info.direct = launch_params.options.direct;
        info.indirect = launch_params.options.indirect;

        uint blocks = (ev->extent.width * ev->extent.height + 255) / 256;
        final_gather <<< blocks, 256 >>> (info);
        CUDA_SYNC_CHECK();
        
        // Update previous view and projection matrices
        launch_params.previous_view = camera.view_matrix(camera_transform);
        launch_params.previous_projection = camera.perspective_matrix();
        launch_params.previous_origin = cuda::to_f3(render_info.camera_transform.position);

        // Report any IO exchanges
        std::string io = optix_io_read(&launch_params.io);
        if (io.size() > 0) {
                std::cout << "Sparse GI output: \"" << io << "\"" << std::endl;
                optix_io_clear(&launch_params.io);
        }
}
