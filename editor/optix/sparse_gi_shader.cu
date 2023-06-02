// Engine headers
#include "include/amadeus/path_tracer.cuh"
#include "include/cuda/core.cuh"
#include "include/cuda/random.cuh"
#include "include/optix/core.cuh"

// Editor headers
#include "sparse_gi_shader.cuh"
#include "../gbuffer_rtx_shader.cuh"

extern "C" {

__constant__ SparseGIParameters parameters;

}

// Aliasing
using namespace kobra;
using namespace kobra::cuda;
using namespace kobra::optix;

struct Packet {
        float3 x;
        float3 n;
        float2 uv;
        bool miss;
        bool entering;
        int64_t id;
};

struct LightInfo {
        float3 position;
        float3 normal;
        float3 emission;
        float area;
        bool sky;
};

// TODO: reservoir buffer and etc
// spatial resampling is done after rendering
__device__
float sample_light(LightInfo &light_info, float3 &seed)
{
        uint total = parameters.area.count + parameters.sky.enabled;
        if (total == 0)
                return -1.0f;

        // Choose light
        uint light_index = cuda::rand_uniform(total, seed);

        if (light_index < parameters.area.count) {
                // Choose triangle
                AreaLight light = parameters.area.lights[light_index];

                uint triangle_index = cuda::rand_uniform(light.triangles, seed);
                uint3 triangle = light.indices[triangle_index];

                glm::vec3 v0 = light.vertices[triangle.x].position;
                glm::vec3 v1 = light.vertices[triangle.y].position;
                glm::vec3 v2 = light.vertices[triangle.z].position;

                v0 = light.model * glm::vec4(v0, 1.0f);
                v1 = light.model * glm::vec4(v1, 1.0f);
                v2 = light.model * glm::vec4(v2, 1.0f);

                glm::vec3 gnormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                // Sample point on triangle
                float3 bary = cuda::pcg3f(seed);

                float u = bary.x;
                float v = bary.y;
                if (u + v > 1.0f) {
                        u = 1.0f - u;
                        v = 1.0f - v;
                }

                glm::vec3 gpoint = v0 * (1.0f - u - v) + v1 * u + v2 * v;
                
                light_info.sky = false;
                light_info.position = make_float3(gpoint.x, gpoint.y, gpoint.z);
                light_info.normal = make_float3(gnormal.x, gnormal.y, gnormal.z);
                light_info.emission = light.emission;
                light_info.area = glm::length(glm::cross(v1 - v0, v2 - v0));
        } else {
                // Environment light, sample direction
                float theta = 2 * PI * cuda::rand_uniform(seed);
                float phi = PI * cuda::rand_uniform(seed);

                float x = sin(phi) * cos(theta);
                float y = cos(phi);
                float z = sin(phi) * sin(theta);

                float3 dir = make_float3(x, y, z);

                light_info.sky = true;
                light_info.position = 1000 * dir;
                light_info.emission = make_float3(sky_at(parameters.sky, dir));
                light_info.area = 4 * PI;
        }

        // Return the pdf of sampling this point light
        return 1.0f / (light_info.area * total);
}

__forceinline__ __device__
float3 cleanse(float3 in)
{
        if (isnan(in.x) || isnan(in.y) || isnan(in.z))
                return make_float3(0.0f);
        return in;
}

__device__
float3 Ld(const SurfaceHit &sh, float3 &seed)
{
        LightInfo light_info;

        float light_pdf = sample_light(light_info, seed);
        if (light_pdf < 0.0f)
                return make_float3(0.0f);

        float3 direction = normalize(light_info.position - sh.x);
        float3 origin = sh.x;
        
        uint i0 = 0;
        uint i1 = 0;

        Packet packet;
        packet.miss = false;

        // TODO: skip if 0 pdf in BRFD sampling...
        pack_pointer(&packet, i0, i1);

        optixTrace(
                parameters.handle,
                origin, direction,
                0.0f, length(light_info.position - origin) - 1e-3f, 0.0f,
                OptixVisibilityMask(0xFF),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT
                | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                0, 1, 0, i0, i1
        );

        float3 out_radiance = sh.mat.emission;
        if (packet.miss && length(out_radiance) < 1e-3f) {
                float3 wi = normalize(light_info.position - origin);
                float R = length(light_info.position - origin);
	
                float3 brdf = cleanse(cuda::brdf(sh, wi, sh.mat.type));

                float ldotn = light_info.sky ? 1.0f : abs(dot(light_info.normal, wi));
                float ndotwi = abs(dot(sh.n, wi));
                float falloff = light_info.sky ? 1.0f : (R * R) + 1e-3f;

                out_radiance = brdf * light_info.emission * ldotn * ndotwi/(light_pdf * falloff);
        }

        return out_radiance;
}

extern "C" __global__ void __raygen__()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();
        const uint2 resolution = parameters.camera.resolution;
      
        int xoffset = idx.x * sizeof(float4);
        int yoffset = (resolution.y - (idx.y + 1));

        float4 raw_position;
        float4 raw_normal;
        float4 raw_uv;

        // TODO: surfaces.position, etc.
        surf2Dread(&raw_position, parameters.position_surface, xoffset, yoffset);
        surf2Dread(&raw_normal, parameters.normal_surface, xoffset, yoffset);
        surf2Dread(&raw_uv, parameters.uv_surface, xoffset, yoffset);

        xoffset = idx.x * sizeof(int32_t);
        yoffset = (resolution.y - (idx.y + 1));

        int32_t raw_index;
        surf2Dread(&raw_index, parameters.index_surface, xoffset, yoffset);

        int32_t triangle_id = raw_index >> 16;
        int32_t material_id = raw_index & 0xFFFF;
        
        int index = idx.x + idx.y * resolution.x;
        if (raw_index == -1) {
                // float3 ray = ray_at(idx);
                // parameters.color[index] = sky_at(ray);
                // // Indicate that there is no valid position here
                // parameters.color[index].w = -1.0f;
                parameters.previous_position[index] = make_float4(0.0f);
                return;
        }

        float3 position = { raw_position.x, raw_position.y, raw_position.z };
        float3 normal = { raw_normal.x, raw_normal.y, raw_normal.z };
        float2 uv = { raw_uv.x, raw_uv.y };

        // Correct the normal
        float3 ray = position - parameters.camera.origin;
        if (dot(ray, normal) > 0.0f)
                normal = -normal;

        float3 seed = make_float3(idx.x, idx.y, parameters.time);
        
        cuda::_material m = parameters.materials[material_id];
        
        SurfaceHit sh;
        sh.x = position;
        sh.wo = normalize(parameters.camera.origin - position);
        sh.n = normalize(normal);
        sh.entering = (raw_normal.w > 0.0f);
        
        convert_material(m, sh.mat, uv);
        float sign = (sh.mat.type == eTransmission) ? -1.0f : 1.0f;
        sh.x += sign * sh.n * 1e-3f;

        // TODO: simoultaenous viewports...
        // TODO: skip direct lighting for highly specular surfaces

        int N = parameters.indirect.N;
        int N2 = N * N;
        uint2 blocks = make_uint2(resolution.x / N, resolution.y / N);
        uint2 coord = make_uint2(idx.x/N, idx.y/N);
        uint block_index = coord.x + coord.y * blocks.x;
        uint offset = parameters.indirect.block_offsets[block_index];
        uint2 local = make_uint2(idx.x % N, idx.y % N);
        uint local_index = local.x + local.y * N;
        offset = (local_index + offset + parameters.counter) % (N2/2);

        // TODO: 2/N2 samples will have secondary bounces, one of which will
        // have third bounces purely for indirect cache

        float3 direct { 0, 0, 0 };
        if (sh.mat.roughness > 0.1f)
                direct = Ld(sh, seed);
               
        float3 indirect { 0, 0, 0 };
        float3 wi;
        float pdf;
        Shading out;

        float3 brdf = eval(sh, wi, pdf, out, seed);
        if (offset == 0 && pdf > 0.0) {
                Packet packet;
                packet.miss = false;
                packet.entering = false;

                uint i0;
                uint i1;

                pack_pointer(&packet, i0, i1);

                optixTrace(
                        parameters.handle,
                        sh.x, wi,
                        0.0f, 1e16f, 0.0f,
                        OptixVisibilityMask(0xFF),
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        0, 1, 0, i0, i1
                );

                if (packet.miss) {
                        indirect = float(N2/2) * make_float3(sky_at(parameters.sky, wi));
                        // indirect = float(N2) * brdf * make_float3(sky_at(wi)) / pdf;
                } else {
                        // TODO: if specular then skip direct lighting and do
                        // another bounce sample
                        cuda::_material m = parameters.materials[packet.id];

                        SurfaceHit sh; 
                        sh.x = packet.x;
                        sh.wo = -wi;
                        sh.n = packet.n;
                        sh.entering = packet.entering;

                        convert_material(m, sh.mat, packet.uv);

                        // indirect = float(N2) * brdf * sh.mat.emission * abs(dot(sh.n, wi)) / pdf;
                        // indirect = float(N2/2) * sh.mat.emission * abs(dot(sh.n, wi));
                        indirect = float(N2/2) * Ld(sh, seed) * abs(dot(sh.n, wi));

                        // TODO: cache irradiance
                }
        }

        // Temporal anti-aliasing of irradiance
        int samples = 0;
        int prev_index = index;
        float3 color { 0, 0, 0 };

        if (parameters.dirty) {
                // TODO: method to return the number of samples
                glm::vec4 p { sh.x.x, sh.x.y, sh.x.z, 1.0f };
                p = parameters.previous_projection * parameters.previous_view * p;

                float u = p.x/p.w;
                float v = p.y/p.w;

                bool in_u_bounds = (u >= -1.0f && u <= 1.0f);
                bool in_v_bounds = (v >= -1.0f && v <= 1.0f);

                if (in_u_bounds && in_v_bounds) {
                        u = (u + 1.0f) * 0.5f;
                        v = (v + 1.0f) * 0.5f;

                        int ix = u * resolution.x;
                        int iy = v * resolution.y;

                        prev_index = iy * resolution.x + ix;
                        float4 prev_color = parameters.indirect.screen_irradiance[prev_index];
                        samples = prev_color.w >= 0 ? prev_color.w : 0;
                }

                // TODO: check validity of temporal accumulation
                // in order to prevent smearing
                if (prev_index >= 0) {
                        float4 prev = parameters.previous_position[prev_index];
                        float3 ppos = { prev.x, prev.y, prev.z };
                        int pmat = *reinterpret_cast <int *> (&prev.w);

                        // TODO: use some heuristic; not every difference needs
                        // to be discarded
                        if (pmat != material_id) {// || length(sh.n - ppos) > 0.1f) {
                                prev_index = index;
                                samples = 0;
                                // color += make_float3(0, 0.5, 0.5);
                        } else {
                                // color += make_float3(0.5, 0.5, 0);
                        }
                }
        } else {
                float4 prev_color = parameters.indirect.screen_irradiance[index];
                samples = prev_color.w >= 0 ? prev_color.w : 0;
        }
        
        // Anti-alias irradiance
        float3 prev_irradiance = make_float3(parameters.indirect.screen_irradiance[prev_index]);
        if (prev_index != index) {
                // Account for differing viewing direction
                float3 cpos = parameters.camera.origin;
                float3 ppos = parameters.previous_origin;

                float3 cdir = normalize(sh.x - cpos);
                float3 pdir = normalize(sh.x - ppos);

                prev_irradiance *= abs(dot(cdir, sh.n))/abs(dot(pdir, sh.n));
        }

        // Explicitly reset samples if necessary
        if (parameters.reset)
                samples = 0;

        float3 new_irradiance = (prev_irradiance * samples + indirect)/(samples + 1.0f);
        parameters.indirect.screen_irradiance[index] = make_float4(cleanse(new_irradiance), samples + 1.0f);

        // Average indirect directions
        int dir_samples = parameters.indirect.direction_samples[index];
        if (parameters.reset)
                dir_samples = 0;

        // Check normal consistency
        float4 prev_direction_bundle = parameters.indirect.irradiance_directions[prev_index];
        float3 prev_direction = make_float3(prev_direction_bundle);
        float prev_pdf_sum = dir_samples * prev_direction_bundle.w;

        float3 new_direction;
        float new_pdf;

        if (parameters.dirty) {
                float pdf_old = cuda::pdf(sh, prev_direction, sh.mat.type);
                float pdf_sum = pdf_old + pdf;
                new_direction = pdf_sum > 0 ? (pdf_old * prev_direction + pdf * wi)/pdf_sum : wi;
                new_direction = normalize(new_direction);
                new_pdf = pdf_sum > 0 ? pdf_sum / (dir_samples + 1.0f) : 0;
        } else {
                float pdf_sum = prev_pdf_sum + pdf;
                new_direction = pdf_sum > 0 ? (prev_pdf_sum * prev_direction + pdf * wi)/pdf_sum : wi;
                new_direction = normalize(new_direction);
                new_pdf = pdf_sum / (dir_samples + 1.0f);
        }

        parameters.indirect.irradiance_directions[index] = make_float4(new_direction, new_pdf);
        parameters.indirect.direction_samples[index] = dir_samples + 1;

        // Save to previous state
        parameters.previous_position[index] = make_float4(sh.n, *reinterpret_cast <float *> (&material_id));
        parameters.direct_lighting[index].data.Le = direct;

        // TODO: for diffuse surfaces, need to modify the direct lighting a
        // little bit to account for the change in view direction

        // TODO: if shadow ray fails (e.g. hits a surface), cache that
        // information and use it to accumulate radiance (or skip new ray
        // altogether...)
}

extern "C" __global__ void __closesthit__()
{
        uint i0 = optixGetPayload_0();
        uint i1 = optixGetPayload_1();

        Packet *packet = unpack_pointer <Packet> (i0, i1);

        ::Hit *hit = (::Hit *) optixGetSbtDataPointer();

        // Indices
        int32_t mat_id = hit->index;
        int32_t tri_id = optixGetPrimitiveIndex();
        packet->id = mat_id;

        // Compute position and normal
        float2 bary = optixGetTriangleBarycentrics();

        float bu = bary.x;
        float bv = bary.y;
        float bw = 1.0f - bu - bv;

        uint3 triangle = hit->triangles[tri_id];
        Vertex v0 = hit->vertices[triangle.x];
        Vertex v1 = hit->vertices[triangle.y];
        Vertex v2 = hit->vertices[triangle.z];

        float2 uv0 = { v0.tex_coords.x, v0.tex_coords.y };
        float2 uv1 = { v1.tex_coords.x, v1.tex_coords.y };
        float2 uv2 = { v2.tex_coords.x, v2.tex_coords.y };
        float2 uv = bw * uv0 + bu * uv1 + bv * uv2;

        packet->uv = { uv.x, 1 - uv.y };

        glm::vec3 glm_pos = bw * v0.position + bu * v1.position + bv * v2.position;
        glm_pos = hit->model * glm::vec4(glm_pos, 1.0f);

        packet->x = { glm_pos.x, glm_pos.y, glm_pos.z };

        // Compute normal
        glm::vec3 e1 = v1.position - v0.position;
        glm::vec3 e2 = v2.position - v0.position;

        e1 = hit->model * glm::vec4(e1, 0.0f);
        e2 = hit->model * glm::vec4(e2, 0.0f);

        glm::vec3 glm_normal = glm::normalize(glm::cross(e1, e2));
        
        float3 ng = { glm_normal.x, glm_normal.y, glm_normal.z };
        float3 wo = optixGetWorldRayDirection();

        if (dot(wo, ng) > 0.0f) {
                glm_normal = -glm_normal;
                packet->entering = false;
        } else {
                packet->entering = true;
        }

        // Shading normal
        glm::vec3 glm_shading_normal = bw * v0.normal + bu * v1.normal + bv * v2.normal;
        glm_shading_normal = hit->model * glm::vec4(glm_shading_normal, 0.0f);

        if (glm::dot(glm_normal, glm_shading_normal) < 0.0f)
                glm_shading_normal = -glm_shading_normal;

        float3 normal = { glm_shading_normal.x, glm_shading_normal.y, glm_shading_normal.z };
        normal = normalize(normal);

        // Transfer to packet
        packet->n = { normal.x, normal.y, normal.z };
}

extern "C" __global__ void __miss__()
{
        uint i0 = optixGetPayload_0();
        uint i1 = optixGetPayload_1();

        Packet *packet = unpack_pointer <Packet> (i0, i1);

        packet->miss = true;
}
