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

static KCUDA_INLINE KCUDA_HOST_DEVICE
void make_ray(uint3 idx, float3 &direction, float3 &seed)
{
	// TODO: use a Blue noise sequence

	// Jittered halton
	seed = make_float3(idx.x, idx.y, 0);

	// int xoff = rand_uniform(parameters.resolution.x, seed);
	// int yoff = rand_uniform(parameters.resolution.y, seed);

        float xoff = rand_uniform(seed) - 0.5f;
        float yoff = rand_uniform(seed) - 0.5f;

	// Compute ray origin and direction
        float2 d = 2.0f * make_float2(
		float(idx.x + xoff)/parameters.resolution.x,
		float(idx.y + yoff)/parameters.resolution.y
	) - 1.0f;

	direction = normalize(d.x * parameters.U + d.y * parameters.V + parameters.W);

        // TODO: generate wavefront of rays
}

extern "C" __global__ void __raygen__()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

        // printf("Tracing ray at %d, %d\n", idx.x, idx.y);
        if (idx.x == 0 && idx.y == 0) {
                optix_io_write_str(&parameters.io, "Passed the first test\n");
        }

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

        cuda::_material m = parameters.materials[material_id];

        float3 position = { raw_position.x, raw_position.y, raw_position.z };
        float3 normal = { raw_normal.x, raw_normal.y, raw_normal.z };
        float2 uv = { raw_uv.x, raw_uv.y };

        float3 color = m.diffuse;
        if (m.textures.has_diffuse)
                color = make_float3(tex2D <float4> (m.textures.diffuse, uv.x, uv.y));

        // Randomly select a light to sample from
        float3 seed = make_float3(idx.x, idx.y, parameters.time);
        uint light_index = cuda::rand_uniform(parameters.area.count, seed);

        AreaLight light = parameters.area.lights[light_index];
        uint triangle_index = cuda::rand_uniform(light.triangles, seed);

        float3 bary = cuda::pcg3f(seed);

        float u = bary.x;
        float v = bary.y;
        if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
        }

        uint3 triangle = light.indices[triangle_index];

        glm::vec3 v0 = light.vertices[triangle.x].position;
        glm::vec3 v1 = light.vertices[triangle.y].position;
        glm::vec3 v2 = light.vertices[triangle.z].position;

        v0 = light.model * glm::vec4(v0, 1.0f);
        v1 = light.model * glm::vec4(v1, 1.0f);
        v2 = light.model * glm::vec4(v2, 1.0f);

        glm::vec3 gpoint = v0 * (1.0f - u - v) + v1 * u + v2 * v;

        glm::vec3 gnormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        // gnormal = glm::normalize(glm::transpose(glm::inverse(light.model)) *
        //                          glm::vec4(gnormal, 0.0f));

        float3 point = { gpoint.x, gpoint.y, gpoint.z };
        float3 nl = { gnormal.x, gnormal.y, gnormal.z };
        float light_area = length(glm::cross(v1 - v0, v2 - v0));
        // TODO: multiply by determinant instead?

        // TODO: get material index for confirmation (already have primitive
        // index...)

        // Ray from point to light
        float3 direction = normalize(point - position);
        float3 origin = position;

        if (isnan(normal.x) || isnan(normal.y) || isnan(normal.z)) {
                origin += direction * 1e-3f;
        } else {
                origin += normal * 1e-3f;
        }
        
        uint i0 = 0;
        uint i1 = 0;

        float hit = 0;
        pack_pointer(&hit, i0, i1);

        optixTrace(
                parameters.handle,
                origin, direction,
                0.0f, length(point - origin) - 1e-3f, 0.0f,
                OptixVisibilityMask(0xFF),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0, 1, 0, i0, i1
        );

        // float3 color
        cuda::Material material;
        material.diffuse = m.diffuse;
        material.specular = m.specular;
        material.emission = m.emission;
        material.roughness = m.roughness;
	material.refraction = m.refraction;
        material.type = m.type;

        if (m.textures.has_diffuse)
                material.diffuse = make_float3(tex2D <float4> (m.textures.diffuse, uv.x, uv.y));

        color = material.emission;
        if (hit == 0 && length(color) < 1e-3f) {
                // TODO: need to explicitly calculate the total number of
                // traingles...
                float light_pdf = 1.0f / (parameters.area.count * light.triangles * light_area);

                float3 wi = normalize(point - position);
                float R = length(point - position);

                SurfaceHit sh;
                sh.x = position;
                sh.wo = normalize(parameters.origin - position);
                sh.n = normalize(normal);
                sh.entering = false;
                sh.mat = material;
	
                float3 brdf = cuda::brdf(sh, wi, eDiffuse);
                color += brdf * light.emission * abs(dot(nl, wi)) * abs(dot(sh.n, wi))/(light_pdf * R * R);
        }

        parameters.color[index] = make_float4(color);
        
        if (idx.x == 0 && idx.y == 0) {
                optix_io_write_str(&parameters.io, "Chose light: ");
                optix_io_write_int(&parameters.io, light_index);
                optix_io_write_str(&parameters.io, "\n");
                optix_io_write_str(&parameters.io, "Chose triangle: ");
                optix_io_write_int(&parameters.io, triangle_index);
                optix_io_write_str(&parameters.io, "/");
                optix_io_write_int(&parameters.io, light.triangles);
                optix_io_write_str(&parameters.io, "\n");
                optix_io_write_str(&parameters.io, "Triangle with ids: ");
                optix_io_write_int(&parameters.io, triangle.x);
                optix_io_write_str(&parameters.io, ", ");
                optix_io_write_int(&parameters.io, triangle.y);
                optix_io_write_str(&parameters.io, ", ");
                optix_io_write_int(&parameters.io, triangle.z);
                optix_io_write_str(&parameters.io, "\n");
                optix_io_write_str(&parameters.io, "shadow hit? ");
                optix_io_write_int(&parameters.io, i0);
                optix_io_write_str(&parameters.io, " or ");
                optix_io_write_int(&parameters.io, hit);
                optix_io_write_str(&parameters.io, "\n");
                optix_io_write_str(&parameters.io, "point: ");
                optix_io_write_int(&parameters.io, point.x * 1000);
                optix_io_write_str(&parameters.io, ", ");
                optix_io_write_int(&parameters.io, point.y * 1000);
                optix_io_write_str(&parameters.io, ", ");
                optix_io_write_int(&parameters.io, point.z * 1000);
        }

        // TODO: if shadow ray fails (e.g. hits a surface), cache that
        // information and use it to accumulate radiance (or skip new ray
        // altogether...)
}

extern "C" __global__ void __closesthit__()
{
        uint i0 = optixGetPayload_0();
        uint i1 = optixGetPayload_1();

        float *hit = unpack_pointer <float> (i0, i1);
        *hit = 1;
}

extern "C" __global__ void __miss__()
{
        uint i0 = optixGetPayload_0();
        uint i1 = optixGetPayload_1();

        float *hit = unpack_pointer <float> (i0, i1);
        *hit = 0;
}
