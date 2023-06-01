#include "editor_viewport.cuh"
#include "include/cuda/cast.cuh"
#include "include/cuda/error.cuh"
#include "include/daemons/material.hpp"
#include "optix/sparse_gi_shader.cuh"

// Gaussian blur of irradiance per pixel
__constant__ float gauss_filter_kernel_5x5[25] = {
        0.00296902,    0.0133062,    0.0219382,    0.0133062,    0.00296902,
        0.0133062,    0.0596343,    0.0983203,    0.0596343,    0.0133062,
        0.0219382,    0.0983203,    0.162103,    0.0983203,    0.0219382,
        0.0133062,    0.0596343,    0.0983203,    0.0596343,    0.0133062,
        0.00296902,    0.0133062,    0.0219382,    0.0133062,    0.00296902
};

__global__
void box_filter(float4 *irradiance, float4 *out, vk::Extent2D extent, cudaSurfaceObject_t normal_surface, int N)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int base_x = index % extent.width;
        int base_y = index / extent.width;

        if (base_x < 0 || base_x >= extent.width ||
            base_y < 0 || base_y >= extent.height)
                return;

        float3 sum = make_float3(0.0f);

        float4 raw_central_normal = make_float4(0.0f);
        surf2Dread(&raw_central_normal, normal_surface, base_x * sizeof(float4), extent.height - (base_y + 1));
        float3 central_normal = make_float3(raw_central_normal);

        int samples = 0;
        for (int i = -N; i <= N; i++) {
                for (int j = -N; j <= N; j++) {
                        int x = base_x + i;
                        int y = base_y + j;

                        if (x < 0 || x >= extent.width ||
                            y < 0 || y >= extent.height)
                                continue;

                        int new_index = x + y * extent.width;
                        float4 raw_normal = make_float4(0.0f);
                        surf2Dread(&raw_normal, normal_surface, x * sizeof(float4), extent.height - (y + 1));
                        float3 normal = make_float3(raw_normal);

                        if (dot(normal, central_normal) < 0.4f)
                                continue;

                        sum += make_float3(irradiance[new_index]);
                        samples++;
                }
        }

        out[index] = samples > 0 ? make_float4(sum/float(samples), 1.0f) : irradiance[index];

        // Copy back to irradiance (except last channel which may be important)
        irradiance[index].x = out[index].x;
        irradiance[index].y = out[index].y;
        irradiance[index].z = out[index].z;

        // TODO: better filter that accounts for normals (bilateral filter...)
}

__global__
void gauss_filter(float4 *irradiance, float4 *out, vk::Extent2D extent)
{
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                        int2 offset = make_int2(i, j);
                        int2 pos = make_int2(x % extent.width, x / extent.width);
                        int2 new_pos = pos + offset;

                        if (new_pos.x < 0 || new_pos.x >= extent.width ||
                                new_pos.y < 0 || new_pos.y >= extent.height)
                                continue;

                        int new_index = new_pos.x + new_pos.y * extent.width;
                        sum += make_float3(irradiance[new_index] * gauss_filter_kernel_5x5[(i + 2) + (j + 2) * 5]);
                }
        }

        out[x] = make_float4(sum, 1.0f);
        
        // Copy back to irradiance (except last channel which may be important)
        irradiance[x].x = out[x].x;
        irradiance[x].y = out[x].y;
        irradiance[x].z = out[x].z;

        // TODO: better filter that accounts for normals (bilateral filter...)
}

struct FinalGatherInfo {
        cuda::_material *materials;
        float time;
        float4 *directions;
        // float3 camera;
        CameraAxis camera;
        float4 *color;

        Reservoir <DirectLightingSample> *direct_lighting;
        float4 *indirect_irradiance;
                
        cudaSurfaceObject_t position_surface;
        cudaSurfaceObject_t normal_surface;
        cudaSurfaceObject_t uv_surface;
        cudaSurfaceObject_t index_surface;

        Sky sky;
        
        vk::Extent2D extent;

        bool direct;
        bool indirect;
};

__device__
float3 ray_at(CameraAxis camera, int x, int y)
{
        y = camera.resolution.y - (y + 1);
        float u = 2.0f * float(x) / float(camera.resolution.x) - 1.0f;
        float v = 2.0f * float(y) / float(camera.resolution.y) - 1.0f;
	return normalize(u * camera.U - v * camera.V + camera.W);
}

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

        surf2Dread(&raw_position, info.position_surface, x * sizeof(float4), info.extent.height - (y + 1));
        surf2Dread(&raw_normal, info.normal_surface, x * sizeof(float4), info.extent.height - (y + 1));
        surf2Dread(&raw_uv, info.uv_surface, x * sizeof(float4), info.extent.height - (y + 1));
        surf2Dread(&raw_index, info.index_surface, x * sizeof(int32_t), info.extent.height - (y + 1));

        // If there is a miss, then exit...
        if (raw_index == -1) {
                float3 ray = ray_at(info.camera, x, y);
                info.color[index] = sky_at(info.sky, ray);
                return;
        }

        int32_t triangle_id = raw_index >> 16;
        int32_t material_id = raw_index & 0xFFFF;

        // Reconstruct the surface hit
        float3 position = { raw_position.x, raw_position.y, raw_position.z };
        float3 normal = { raw_normal.x, raw_normal.y, raw_normal.z };
        float2 uv = { raw_uv.x, 1 - raw_uv.y };

        // Correct the normal
        float3 ray = position - info.camera.origin;
        if (dot(ray, normal) > 0.0f)
                normal = -normal;

        float3 seed = make_float3(x, y, info.time);

        cuda::_material m = info.materials[material_id];

        cuda::SurfaceHit sh;
        sh.x = position;
        sh.wo = normalize(info.camera.origin - position);
        sh.n = normalize(normal);
        sh.entering = (raw_normal.w > 0.0f);

        convert_material(m, sh.mat, uv);
        float sign = (sh.mat.type == eTransmission) ? -1.0f : 1.0f;
        sh.x += sign * sh.n * 1e-3f;

        // Get brdf value
        // TODO: get weighted average of directions (based on pdfs)
        float3 wi = make_float3(info.directions[index]);
        float3 brdf = cuda::brdf(sh, wi, eDiffuse);
        float pdf = cuda::pdf(sh, wi, eDiffuse);

        float3 indirect = pdf > 0.0 ? brdf * make_float3(info.indirect_irradiance[index])/pdf : make_float3(0.0f);
        // float3 indirect = make_float3(info.indirect_irradiance[index]);
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
                
                if (launch_params.indirect.direction_samples != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.indirect.direction_samples));

                if (launch_params.indirect.final_irradiance != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.indirect.final_irradiance));
                
                if (launch_params.direct_lighting != 0)
                        CUDA_CHECK(cudaFree((void *) launch_params.direct_lighting));

                int size = new_extent.width * new_extent.height;
                CUDA_CHECK(cudaMalloc((void **) &launch_params.previous_position, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.screen_irradiance, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.final_irradiance, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.irradiance_directions, size * sizeof(float4)));
                CUDA_CHECK(cudaMalloc((void **) &launch_params.indirect.direction_samples, size * sizeof(float)));
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

        // Configure camera axis
        auto uvw = uvw_frame(camera, camera_transform);

        launch_params.camera.U = cuda::to_f3(uvw.u);
        launch_params.camera.V = cuda::to_f3(uvw.v);
        launch_params.camera.W = cuda::to_f3(uvw.w);
        launch_params.camera.origin = cuda::to_f3(render_info.camera_transform.position);
        launch_params.camera.resolution = { ev->extent.width, ev->extent.height };

        // Configure surfaces
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
        uint blocks = (ev->extent.width * ev->extent.height + 255) / 256;
        gauss_filter <<< blocks, 256 >>> (
                launch_params.indirect.screen_irradiance,
                launch_params.indirect.final_irradiance,
                ev->extent // , launch_params.normal_surface
        );
                

        FinalGatherInfo info;
        info.directions = launch_params.indirect.irradiance_directions;
        info.camera = launch_params.camera;
        info.color = ev->common_rtx.dev_color;
        info.direct_lighting = launch_params.direct_lighting;
        info.index_surface = launch_params.index_surface;
        info.indirect_irradiance = launch_params.indirect.final_irradiance;
        info.materials = (cuda::_material *) ev->common_rtx.dev_materials;
        info.normal_surface = launch_params.normal_surface;
        info.position_surface = launch_params.position_surface;
        info.time = launch_params.time;
        info.uv_surface = launch_params.uv_surface;
        info.sky.texture = ev->environment_map.texture;
        info.sky.enabled = ev->environment_map.valid;
        info.extent = ev->extent;
        info.direct = launch_params.options.direct;
        info.indirect = launch_params.options.indirect;

        blocks = (ev->extent.width * ev->extent.height + 255) / 256;
        final_gather <<< blocks, 256 >>> (info);
        CUDA_SYNC_CHECK();
        
        // Update previous view and projection matrices
        launch_params.previous_view = camera.view_matrix(camera_transform);
        launch_params.previous_projection = camera.perspective_matrix();
        launch_params.previous_origin = cuda::to_f3(render_info.camera_transform.position);

        // Report any IO exchanges
        // std::string io = optix_io_read(&launch_params.io);
        // if (io.size() > 0) {
        //         std::cout << "Sparse GI output: \"" << io << "\"" << std::endl;
        //         optix_io_clear(&launch_params.io);
        // }
}
