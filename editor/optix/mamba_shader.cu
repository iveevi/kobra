// Engine headers
#include "include/cuda/brdf.cuh"
#include "include/cuda/core.cuh"
#include "include/cuda/random.cuh"
#include "include/optix/core.cuh"
#include "include/optix/sbt.cuh"

// Editor headers
#include "mamba_shader.cuh"

extern "C" {

__constant__ MambaLaunchInfo info;

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

__forceinline__ __device__
float luminance(float3 in)
{
        return in.x + in.y + in.z;
        // return 0.2126f * in.x + 0.7152f * in.y + 0.0722f * in.z;
}

__forceinline__ __device__
bool occluded(float3 origin, float3 target)
{
        static const float epsilon = 1e-3f;

        Packet packet;
        packet.miss = false;

        unsigned int i0 = 0;
        unsigned int i1 = 0;

        pack_pointer(&packet, i0, i1);

        float3 direction = normalize(target - origin);
        optixTrace(
                info.handle,
                origin, direction,
                0.0f, length(target - origin) - epsilon, 0.0f,
                OptixVisibilityMask(0xFF),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT
                | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                0, 1, 0, i0, i1
        );

        return !packet.miss;
}

__forceinline__ __device__
float3 cleanse(float3 in)
{
        if (isnan(in.x) || isnan(in.y) || isnan(in.z))
                return make_float3(0.0f);
        return in;
}

__device__ __forceinline__
float3 direct_unoccluded(const SurfaceHit &sh, const LightInfo &li)
{
        float3 wi = normalize(li.position - sh.x);
        float R = length(li.position - sh.x);

	// Assume that the light is visible
	float3 rho = cleanse(brdf(sh, wi, sh.mat.type));
        float ldotn = li.sky ? 1.0f : abs(dot(li.normal, wi));
        float ndotwi = abs(dot(sh.n, wi));
        float falloff = li.sky ? 1.0f : 1.0f/(R * R);

        return rho * li.emission * ldotn * ndotwi * falloff;
}

__device__
float3 ray_at(uint3 idx)
{
        const CameraAxis &axis = info.camera;
        idx.y = axis.resolution.y - (idx.y + 1);
        float u = 2.0f * float(idx.x) / float(axis.resolution.x) - 1.0f;
        float v = 2.0f * float(idx.y) / float(axis.resolution.y) - 1.0f;
	return normalize(u * axis.U - v * axis.V + axis.W);
}

__device__
float sample_light(LightInfo &light_info, float3 &seed)
{
        constexpr float ENVIRONMENT_DISTANCE = 1000.0f;

        uint total = info.area.count + info.sky.enabled;
        if (total == 0)
                return -1.0f;

        // Choose light
        uint light_index = cuda::rand_uniform(total, seed);

        if (light_index < info.area.count) {
                // Choose triangle
                AreaLight light = info.area.lights[light_index];

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
                light_info.position = ENVIRONMENT_DISTANCE * dir;
                light_info.emission = make_float3(sky_at(info.sky, dir));
                light_info.area = 4 * PI;
        }

        // Return the pdf of sampling this point light
        return 1.0f/(light_info.area * total);
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
                info.handle,
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

__device__
bool load_surface_hit(SurfaceHit &sh, const uint3 &idx)
{
        const uint2 resolution = info.camera.resolution;

        // Read G-buffer data
        float4 raw_position;
        float4 raw_normal;
        float4 raw_uv;
        
        int xoffset = idx.x * sizeof(float4);
        int yoffset = (resolution.y - (idx.y + 1));
        
        surf2Dread(&raw_position, info.position, xoffset, yoffset);
        surf2Dread(&raw_normal, info.normal, xoffset, yoffset);
        surf2Dread(&raw_uv, info.uv, xoffset, yoffset);

        xoffset = idx.x * sizeof(int32_t);
        yoffset = (resolution.y - (idx.y + 1));

        int32_t raw_index;
        surf2Dread(&raw_index, info.index, xoffset, yoffset);

        int32_t triangle_id = raw_index >> 16;
        int32_t material_id = raw_index & 0xFFFF;
       
        if (raw_index == -1)
                return false;

        // Reconstruct surface properties
        float3 position = { raw_position.x, raw_position.y, raw_position.z };
        float3 normal = { raw_normal.x, raw_normal.y, raw_normal.z };
        float2 uv = { raw_uv.x, raw_uv.y };

        // Correct the normal
        float3 ray = position - info.camera.origin;
        if (dot(ray, normal) > 0.0f)
                normal = -normal;

        // SurfaceHit sh;
        sh.x = position;
        sh.wo = normalize(info.camera.origin - position);
        sh.n = normalize(normal);
        sh.entering = (raw_normal.w > 0.0f);
        
        cuda::_material m = info.materials[material_id];
        convert_material(m, sh.mat, uv);

        float sign = (sh.mat.type == eTransmission) ? -1.0f : 1.0f;
        sh.x += sign * sh.n * 1e-3f;

        return true;
}

// Direct lighting ray generation kernel
extern "C" __global__ void __raygen__direct_primary()
{
        auto direct = info.direct;

	// Compute coordinates
	const uint3 idx = optixGetLaunchIndex();
        const uint2 resolution = info.camera.resolution;
        int index = idx.x + idx.y * resolution.x;

        // if (idx.x + idx.y == 0) {
        //         optix_io_write_str(&info.io, "Hiii, from the raygen! @");
        //         optix_io_write_int(&info.io, idx.x);
        //         optix_io_write_str(&info.io, ", ");
        //         optix_io_write_int(&info.io, idx.y);
        // }

        direct.reservoirs[index].reset();
        SurfaceHit sh;
        if (!load_surface_hit(sh, idx))
                return;

        // Sample direct lighting
        float3 seed = make_float3(idx.x, idx.y, info.time);
        for (int i = 0; i < 32; i++) {
                LightInfo lighting;
                float pdf = sample_light(lighting, seed);

                float3 Ld = direct_unoccluded(sh, lighting);
                float target = Ld.x + Ld.y + Ld.z;

                direct.reservoirs[index].update(lighting, target/pdf > 0.0f ? target/pdf : 0.0f);
        }

        // TODO: profiler using cuda events...
        LightInfo resampled = direct.reservoirs[index].data;
        float3 Ld_resampled = direct_unoccluded(sh, resampled);
        bool is_occluded = occluded(sh.x, resampled.position);
        direct.reservoirs[index].resample(luminance(Ld_resampled * (1.0f - is_occluded)));
        // float3 new_direct = Ld_resampled * (1.0f - is_occluded) * direct.reservoirs[index].W;
        // float3 new_direct = pdf > 0 ? Ld_resampled * (1.0f - is_occluded)/pdf : make_float3(0.0f);

        // int samples = info.samples;
        // direct.Le[index] = (direct.Le[index] * samples + sh.mat.emission + new_direct)/(samples + 1);
        // direct.Le[index] = make_float3(0,1,0);
}

// Calculate target function for reservoir
__forceinline__ __device__
float target(const SurfaceHit &sh, const Reservoir <LightInfo> &reservoir)
{
	const LightInfo &li = reservoir.data;
        return luminance(direct_unoccluded(sh, li));
}

__forceinline__ __device__
float occluded_target(const SurfaceHit &hit, const Reservoir <LightInfo> &reservoir)
{
        const LightInfo &li = reservoir.data;
        float3 Li = direct_unoccluded(hit, li);
        return luminance(Li * (1.0f - occluded(hit.x, li.position)));
}

__forceinline__ __device__
int reproject(int index, glm::vec3 position)
{
	int pindex = -1;
	if (!info.dirty)
		return index;

	// Project position
	glm::vec4 p = info.previous_projection
		* info.previous_view
		* glm::vec4(position, 1.0f);

	float u = p.x/p.w;
	float v = p.y/p.w;

	bool in_u_bounds = (u >= -1.0f && u <= 1.0f);
	bool in_v_bounds = (v >= -1.0f && v <= 1.0f);

	if (in_u_bounds && in_v_bounds) {
                uint2 resolution = info.camera.resolution;

		u = (u + 1.0f) * 0.5f;
		v = (v + 1.0f) * 0.5f;

		int ix = u * resolution.x + 0.5;
		int iy = v * resolution.y + 0.5;

		pindex = iy * resolution.x + ix;
	}

	return pindex;
}

// Temporal resampling
extern "C" __global__ void __raygen__temporal_reuse()
{
        auto direct = info.direct;

	// Compute coordinates
	const uint3 idx = optixGetLaunchIndex();
        const uint2 resolution = info.camera.resolution;
        int index = idx.x + idx.y * resolution.x;

        // if (idx.x + idx.y == 0) {
        //         optix_io_write_str(&info.io, "Hiii, from the (temporal) raygen! @");
        //         optix_io_write_int(&info.io, idx.x);
        //         optix_io_write_str(&info.io, ", ");
        //         optix_io_write_int(&info.io, idx.y);
        // }

        SurfaceHit sh;
        if (!load_surface_hit(sh, idx)) {
                direct.Le[index] = make_float3(sky_at(info.sky, ray_at(idx)));
                direct.previous[index].reset();
                return;
        }

        // Temporal resampling
        int pindex = reproject(index, { sh.x.x, sh.x.y, sh.x.z });

        Reservoir <LightInfo> *current = &direct.reservoirs[index];
        Reservoir <LightInfo> *previous = nullptr;
        if (pindex >= 0)
                previous = &direct.reservoirs[pindex];

        if (current->size() <= 0) {
                // direct.Le[index] = make_float3(0,0,1);
                // direct.previous[index].reset();
                return;
        }

        // TODO: reproject to get previous position
        float3 seed = make_float3(idx.x, idx.y, info.time);
        Reservoir <LightInfo> merged(2.0 * seed);
	
        float t_current = occluded_target(sh, *current);
        merged.update(current->data, t_current * current->W * current->M);

        int N = current->M;
        if (previous && previous->size() > 0) {
                // M-capping
                previous->M = min(previous->M, 200);

                float t_previous = occluded_target(sh, *previous);
                merged.update(previous->data, t_previous * previous->W * previous->M);
                N += (t_previous > 0) * previous->M;
        }

        // Resample merged reservoir
        merged.M = N;
        float t_merged = target(sh, merged);
        merged.resample(t_merged);
        *current = merged;
        
        // LightInfo resampled = current->data;
        // float3 Ld_resampled = direct_unoccluded(sh, resampled);
        // bool is_occluded = occluded(sh.x, resampled.position);
        // float3 new_direct = Ld_resampled * (1.0f - is_occluded) * current->W;
        // int samples = info.samples;
        // direct.Le[index] = (direct.Le[index] * samples + sh.mat.emission + new_direct)/(samples + 1);
        // direct.previous[index] = *current;
        // TODO: or memcpy previous reservoirs?
}

__forceinline__ __device__
int sample_spatial_neighborhood(int index, float3 &seed, int radius = 20)
{
        uint2 resolution = info.camera.resolution;
	int width = resolution.x;
	int height = resolution.y;

	int x = index % width;
	int y = index / width;

	float3 offset = rand_uniform_3f(seed);
	float theta = 2.0f * M_PI * offset.x;
	float r = radius * sqrt(offset.y);

	int x0 = x + r * cos(theta);
	int y0 = y + r * sin(theta);

	x0 = clamp(x0, 0, width - 1);
	y0 = clamp(y0, 0, height - 1);

	return x0 + y0 * width;
}

// Spatial resampling
extern "C" __global__ void __raygen__spatial_reuse()
{
        auto direct = info.direct;

	// Compute coordinates
	const uint3 idx = optixGetLaunchIndex();
        const uint2 resolution = info.camera.resolution;
        int index = idx.x + idx.y * resolution.x;

        // if (idx.x + idx.y == 0) {
        //         optix_io_write_str(&info.io, "Hiii, from the (temporal) raygen! @");
        //         optix_io_write_int(&info.io, idx.x);
        //         optix_io_write_str(&info.io, ", ");
        //         optix_io_write_int(&info.io, idx.y);
        // }

        SurfaceHit sh;
        if (!load_surface_hit(sh, idx)) {
                direct.Le[index] = make_float3(sky_at(info.sky, ray_at(idx)));
                direct.previous[index].reset();
                return;
        }

        // Spatial resampling
        Reservoir <LightInfo> *current = &direct.reservoirs[index];
        if (current->size() <= 0) {
                direct.Le[index] = make_float3(0,0,1);
                direct.previous[index].reset();
                return;
        }
		
        float3 seed = make_float3(idx.x, idx.y, info.time);
        Reservoir <LightInfo> merged(2.0 * seed);

        float t_current = target(sh, *current);
        merged.update(current->data, t_current * current->W * current->M);

        // Sample spatial neighborhood
        constexpr int SAMPLES = 5;

        SurfaceHit hits[SAMPLES];
        int sizes[SAMPLES];

        int count = 0;
        for (int i = 0; i < SAMPLES; i++) {
                int index0 = sample_spatial_neighborhood(index, merged.seed, 30);
                uint3 idx0 = make_uint3(index0 % resolution.x, index0/resolution.x, 1);

                Reservoir <LightInfo> neighbor = info.direct.reservoirs[index0];
                float t_neighbor = occluded_target(sh, neighbor);

                if (neighbor.size() > 0)
                        merged.update(neighbor.data, t_neighbor * neighbor.W * neighbor.M);

                // Reconstruct neighbor surface intersection information
                // Material material0 = parameters.materials_buffer[index0];
                // glm::vec3 position0 = parameters.buffers.position[index0];
                // glm::vec3 normal0 = parameters.buffers.normal[index0];
                // SurfaceHit sh0;
                load_surface_hit(hits[i], idx0);

                // hits[i] = sh0;
                sizes[i] = neighbor.size();

                // Add to count
                count += neighbor.M;
        }

        merged.M = current->M + count;

        // Fix bias (Original ReSTIR paper)
        int Z = current->M;
        for (int i = 0; i < SAMPLES; i++) {
                float t_neighbor = (hits[i].entering) ? occluded_target(hits[i], merged) : 0.0f;
                if (t_neighbor > 0.0f)
                        Z += sizes[i];
        }

        // Resample merged reservoir
        // TODO: option for bias merged instead
        float t_merged = target(sh, merged);
        float denominator = t_merged * Z;
        merged.W = (denominator > 0.0f) ? merged.w/denominator : 0.0f;
        *current = merged;
        
        // Compute shading
        LightInfo resampled = current->data;
        float3 Ld_resampled = direct_unoccluded(sh, resampled);
        bool is_occluded = occluded(sh.x, resampled.position);
        float3 new_direct = Ld_resampled * (1.0f - is_occluded) * current->W;
        int samples = info.samples;
        direct.Le[index] = (direct.Le[index] * samples + sh.mat.emission + new_direct)/(samples + 1);
        direct.previous[index] = *current;
}

// Closest hit for indirect rays
extern "C" __global__ void __closesthit__()
{
        uint i0 = optixGetPayload_0();
        uint i1 = optixGetPayload_1();

        Packet *packet = unpack_pointer <Packet> (i0, i1);

        // TODO: rewrite this somehow...
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
