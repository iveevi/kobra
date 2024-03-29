// Engine headers
#include "include/cuda/core.cuh"
#include "include/cuda/random.cuh"
#include "include/optix/core.cuh"

// Editor headers
#include "gbuffer_rtx_shader.cuh"

extern "C" {

__constant__ GBufferParameters parameters;

}

// Aliasing
using namespace kobra;
using namespace kobra::cuda;
using namespace kobra::optix;

static KCUDA_INLINE KCUDA_HOST_DEVICE
void make_ray(uint3 idx, float3 &direction, float3 &seed)
{
	// TODO: use a Blue noise sequence

	// Jittered halton
	seed = make_float3(idx.x, idx.y, 0);

	// int xoff = rand_uniform(parameters.resolution.x, seed);
	// int yoff = rand_uniform(parameters.resolution.y, seed);

        // float xoff = rand_uniform(seed) - 0.5f;
        // float yoff = rand_uniform(seed) - 0.5f;
        float xoff = 0.0f, yoff = 0.0f;

	// Compute ray origin and direction
        float2 d = 2.0f * make_float2(
		float(idx.x + xoff)/parameters.resolution.x,
		float(idx.y + yoff)/parameters.resolution.y
	) - 1.0f;

	direction = normalize(d.x * parameters.U - d.y * parameters.V + parameters.W);

        // TODO: generate wavefront of rays
}

struct Packet {
        float4 position;
        float4 normal;
        float4 uv;
        int32_t id;
};

extern "C" __global__ void __raygen__()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

        // printf("Tracing ray at %d, %d\n", idx.x, idx.y);
        if (idx.x == 0 && idx.y == 0) {
                optix_io_write_str(&parameters.io, "Passed the first test\n");
        }

        float3 direction;
        float3 origin = parameters.origin;
        float3 seed;

        make_ray(idx, direction, seed);

        Packet packet;

        packet.position = { 0.0f, 0.0f, 0.0f, 0.0f };
        packet.normal = { 0.0f, 0.0f, 0.0f, 0.0f };
        packet.uv = { 0.0f, 0.0f, 0.0f, 0.0f };
        packet.id = -1;
 
	unsigned int i0, i1;
	pack_pointer(&packet, i0, i1);

        optixTrace(
                parameters.handle,
                origin, direction,
                0.0f, 1e16f, 0.0f,
                OptixVisibilityMask(0xFF),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0, 1, 0,
                i0, i1
        );
        
        surf2Dwrite(packet.position, parameters.position_surface, idx.x * sizeof(float4), idx.y);
        surf2Dwrite(packet.normal, parameters.normal_surface, idx.x * sizeof(float4), idx.y);
        surf2Dwrite(packet.uv, parameters.uv_surface, idx.x * sizeof(float4), idx.y);
        surf2Dwrite(packet.id, parameters.index_surface, idx.x * sizeof(int32_t), idx.y);
}

extern "C" __global__ void __closesthit__()
{
        // Get information
        Packet *packet;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	packet = unpack_pointer <Packet> (i0, i1);
        
        Hit *hit = (Hit *) optixGetSbtDataPointer();

        // Indices
        int32_t mat_id = hit->index;
        int32_t tri_id = optixGetPrimitiveIndex();

        packet->id = mat_id | (tri_id << 16);

        // Compute position and normal
        // TODO: UV as well?
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

        packet->uv = { uv.x, 1 - uv.y, 0.0f, 0.0f };

        glm::vec3 glm_pos = bw * v0.position + bu * v1.position + bv * v2.position;
        glm_pos = hit->model * glm::vec4(glm_pos, 1.0f);

        packet->position = { glm_pos.x, glm_pos.y, glm_pos.z, 1.0f };

        // Compute normal
        glm::vec3 e1 = v1.position - v0.position;
        glm::vec3 e2 = v2.position - v0.position;

        e1 = hit->model * glm::vec4(e1, 0.0f);
        e2 = hit->model * glm::vec4(e2, 0.0f);

        glm::vec3 glm_normal = glm::normalize(glm::cross(e1, e2));

        float3 ng = { glm_normal.x, glm_normal.y, glm_normal.z };
        float3 wo = optixGetWorldRayDirection();

        bool entering = false;
        if (dot(wo, ng) > 0.0f) {
                glm_normal = -glm_normal;
                entering = false;
        } else {
                entering = true;
        }

        // Shading normal
        glm::vec3 glm_shading_normal = bw * v0.normal + bu * v1.normal + bv * v2.normal;
        glm_shading_normal = hit->model * glm::vec4(glm_shading_normal, 0.0f);

        if (glm::dot(glm_normal, glm_shading_normal) < 0.0f)
                glm_shading_normal = -glm_shading_normal;

        float3 normal = { glm_shading_normal.x, glm_shading_normal.y, glm_shading_normal.z };
        normal = normalize(normal);

        // Transfer to packet
        packet->normal = { normal.x, normal.y, normal.z, (float) entering };
}

extern "C" __global__ void __miss__() {}
