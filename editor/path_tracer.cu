// Engine headers
#include "include/amadeus/path_tracer.cuh"
#include "include/cuda/core.cuh"
#include "include/cuda/random.cuh"
#include "include/optix/core.cuh"

// Editor headers
#include "path_tracer.cuh"

extern "C" {

__constant__ PathTracerParameters parameters;

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
};

__device__
void make_ray(uint3 idx, float3 &direction, float3 &seed)
{
	// Jittered halton
	seed = make_float3(idx.x, idx.y, 0);

        float xoff = rand_uniform(seed) - 0.5f;
        float yoff = rand_uniform(seed) - 0.5f;

	// Compute ray origin and direction
        idx.y = parameters.resolution.y - (idx.y + 1);

        float2 d = 2.0f * make_float2(
		float(idx.x + xoff)/parameters.resolution.x,
		float(idx.y + yoff)/parameters.resolution.y
	) - 1.0f;

	direction = normalize(d.x * parameters.U - d.y * parameters.V + parameters.W);
}

__device__
float sample_light(LightInfo &light_info, float3 &seed)
{
        // Choose light
        uint light_index = cuda::rand_uniform(parameters.area.count, seed);

        // Choose triangle
        AreaLight light = parameters.area.lights[light_index];
        light_info.emission = light.emission;

        uint triangle_index = cuda::rand_uniform(light.triangles, seed);
        uint3 triangle = light.indices[triangle_index];

        glm::vec3 v0 = light.vertices[triangle.x].position;
        glm::vec3 v1 = light.vertices[triangle.y].position;
        glm::vec3 v2 = light.vertices[triangle.z].position;

        v0 = light.model * glm::vec4(v0, 1.0f);
        v1 = light.model * glm::vec4(v1, 1.0f);
        v2 = light.model * glm::vec4(v2, 1.0f);

        glm::vec3 gnormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        light_info.normal = make_float3(gnormal.x, gnormal.y, gnormal.z);
        light_info.area = glm::length(glm::cross(v1 - v0, v2 - v0));

        // Sample point on triangle
        float3 bary = cuda::pcg3f(seed);

        float u = bary.x;
        float v = bary.y;
        if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
        }

        glm::vec3 gpoint = v0 * (1.0f - u - v) + v1 * u + v2 * v;
        light_info.position = make_float3(gpoint.x, gpoint.y, gpoint.z);

        // Return the pdf of sampling this point light
        return 1.0f / (light_info.area * parameters.area.count);
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

__device__
float3 radiance(const SurfaceHit &sh, float3 &seed, int depth)
{
        LightInfo light_info;
        float light_pdf = sample_light(light_info, seed);

        float3 direction = normalize(light_info.position - sh.x);
        float3 origin = sh.x;

        float sign = (sh.mat.type == eTransmission) ? -1.0f : 1.0f;
        if (isnan(sh.n.x) || isnan(sh.n.y) || isnan(sh.n.z))
                origin += sign * direction * 1e-3f;
        else
                origin += sign * sh.n * 1e-3f;
        
        uint i0 = 0;
        uint i1 = 0;

        Packet packet;
        packet.miss = false;

        pack_pointer(&packet, i0, i1);

        optixTrace(
                parameters.handle,
                origin, direction,
                0.0f, length(light_info.position - origin) - 1e-3f, 0.0f,
                OptixVisibilityMask(0xFF),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT
                | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                // TODO: use miss instead of hit and disable closest hit, etc
                0, 1, 0, i0, i1
        );

        float3 out_radiance = sh.mat.emission;
        if (packet.miss && length(out_radiance) < 1e-3f) {
                float3 wi = normalize(light_info.position - sh.x);
                float R = length(light_info.position - sh.x);
	
                float3 brdf = cuda::brdf(sh, wi, sh.mat.type);
                out_radiance = brdf * light_info.emission * abs(dot(light_info.normal, wi)) * abs(dot(sh.n, wi))/(light_pdf * R * R);
        }

        return out_radiance;
}

extern "C" __global__ void __raygen__()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();
       
        float4 raw_position;
        surf2Dread(&raw_position, parameters.position_surface, idx.x * sizeof(float4),
                parameters.resolution.y - (idx.y + 1));
        
        float4 raw_normal;
        surf2Dread(&raw_normal, parameters.normal_surface, idx.x * sizeof(float4),
                parameters.resolution.y - (idx.y + 1));

        float4 raw_uv;
        surf2Dread(&raw_uv, parameters.uv_surface, idx.x * sizeof(float4),
                parameters.resolution.y - (idx.y + 1));

        int32_t raw_index;
        surf2Dread(&raw_index, parameters.index_surface, idx.x * sizeof(int32_t),
                parameters.resolution.y - (idx.y + 1));

        int32_t triangle_id = raw_index >> 16;
        int32_t material_id = raw_index & 0xFFFF;
        
        int index = idx.x + idx.y * parameters.resolution.x;
        if (raw_index == -1) {
                parameters.color[index] = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
                return;
        }

        float3 position = { raw_position.x, raw_position.y, raw_position.z };
        float3 normal = { raw_normal.x, raw_normal.y, raw_normal.z };
        float2 uv = { raw_uv.x, raw_uv.y };

        // Correct the normal
        float3 ray = position - parameters.origin;
        if (dot(ray, normal) > 0.0f)
                normal = -normal;

        float3 seed = make_float3(idx.x, idx.y, parameters.time);
        
        cuda::_material m = parameters.materials[material_id];
        
        SurfaceHit sh;
        sh.x = position; // TODO: offset...
        sh.wo = normalize(parameters.origin - position);
        sh.n = normalize(normal);
        sh.entering = false;

        convert_material(m, sh.mat, uv);

        // float3 color = radiance(sh, seed, 0);
        float3 color = make_float3(0.0f);
        float3 beta = make_float3(1.0f);

        static constexpr int MAX_DEPTH = 8;
        for (int depth = 0; depth < MAX_DEPTH; depth++) {
                color += beta * radiance(sh, seed, depth);
                
                float3 wi;
                float pdf;
                Shading out;

                float3 brdf = eval(sh, wi, pdf, out, seed);
                if (pdf > 0.0 && depth < MAX_DEPTH - 1) {
                        Packet packet;
                        packet.miss = false;
                        packet.entering = false;

                        float sign = (sh.mat.type == eTransmission) ? -1.0f : 1.0f;
                        float3 origin = sh.x + sign * sh.n * 1e-3f;

                        uint i0;
                        uint i1;

                        pack_pointer(&packet, i0, i1);

                        optixTrace(
                                parameters.handle,
                                origin, wi,
                                0.0f, 1e16f, 0.0f,
                                OptixVisibilityMask(0xFF),
                                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                                // TODO: use miss instead of hit and disable closest hit, etc
                                0, 1, 0, i0, i1
                        );

                        if (!packet.miss) {
                                cuda::_material m = parameters.materials[packet.id];
                                
                                sh.x = packet.x;
                                sh.wo = -wi;
                                sh.n = packet.n;
                                sh.entering = false;

                                convert_material(m, sh.mat, packet.uv);

                                beta *= brdf * abs(dot(sh.n, wi)) / pdf;
                        } else {
                                break;
                        }
                } else {
                        break;
                }
        }

        // Store color
        parameters.color[index] = make_float4(color);

        // TODO: if shadow ray fails (e.g. hits a surface), cache that
        // information and use it to accumulate radiance (or skip new ray
        // altogether...)
}

extern "C" __global__ void __closesthit__()
{
        uint i0 = optixGetPayload_0();
        uint i1 = optixGetPayload_1();

        Packet *packet = unpack_pointer <Packet> (i0, i1);

        // packet->miss = false;
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

        packet->uv = { uv.x, uv.y };

        glm::vec3 glm_pos = bw * v0.position + bu * v1.position + bv * v2.position;
        glm_pos = hit->model * glm::vec4(glm_pos, 1.0f);

        packet->x = { glm_pos.x, glm_pos.y, glm_pos.z };

        // Compute normal
        glm::vec3 e1 = v1.position - v0.position;
        glm::vec3 e2 = v2.position - v0.position;

        e1 = hit->model * glm::vec4(e1, 0.0f);
        e2 = hit->model * glm::vec4(e2, 0.0f);

        glm::vec3 glm_normal = glm::normalize(glm::cross(e1, e2));

        // Shading normal
        glm::vec3 glm_shading_normal = bw * v0.normal + bu * v1.normal + bv * v2.normal;
        glm_shading_normal = hit->model * glm::vec4(glm_shading_normal, 0.0f);

        if (glm::dot(glm_normal, glm_shading_normal) < 0.0f)
                glm_shading_normal = -glm_shading_normal;

        float3 normal = { glm_shading_normal.x, glm_shading_normal.y, glm_shading_normal.z };
        normal = normalize(normal);

        float3 wo = optixGetWorldRayDirection();
        if (dot(wo, normal) > 0.0f) {
                normal = -normal;
                packet->entering = true;
        }

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
