#version 450

#include "random.glsl"
#include "ggx.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)
uniform writeonly image2D framebuffer;

layout (binding = 1)
uniform sampler2D environment;

layout (push_constant) uniform PushConstants {
        // Camera
        vec3 origin;

        // Material
        vec3 diffuse;
        vec3 specular;
        float roughness;
} push_constants;

// Check if ray hits unit sphere in the center
bool hit_sphere(vec3 origin, vec3 direction, out vec3 x, out vec3 n)
{
        vec3 center = vec3(0, 0, 0);
        float radius = 1.0;

        vec3 oc = origin - center;
        float a = dot(direction, direction);
        float b = 2.0 * dot(oc, direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0 * a * c;

        if (discriminant < 0.0)
                return false;

        float t = (-b - sqrt(discriminant)) / (2.0 * a);
        x = origin + t * direction;
        n = normalize(x - center);

        return true;
}

// Sample the environment map
vec4 env_radiance(vec3 direction)
{
        float phi = atan(direction.z, direction.x);
        float theta = acos(direction.y);

        vec2 env_uv = vec2(phi, theta)/vec2(2.0 * PI, PI);
        return texture(environment, env_uv);
}

void main()
{
        ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
        ivec2 size = imageSize(framebuffer);

        vec2 uv = (vec2(coord) + vec2(0.0)) / vec2(size);
        uv = uv * 2.0 - 1.0;

        // Compute the ray direction
        float fov = 45.0;
        fov = fov * PI / 180.0;

        float h = tan(fov / 2.0);
        float aspect = 1.0f;

        float height = 2.0 * h;
        float width = aspect * height;

        vec3 w = normalize(push_constants.origin - vec3(0.0));
        vec3 u = normalize(cross(vec3(0, 1, 0), w));
        vec3 v = normalize(cross(w, u));

        vec3 direction = normalize(uv.x * u + uv.y * v - w);

        // Compute UV on the environment map
        vec3 x;
        vec3 n;

        vec4 radiance;
        if (hit_sphere(push_constants.origin, direction, x, n)) {
                // Seed
                vec3 seed = vec3(coord, 0.0);

                Material material;
                material.diffuse = push_constants.diffuse;
                material.specular = push_constants.specular;
                material.roughness = push_constants.roughness;
                
                vec3 wo = -direction;
                vec3 color = vec3(0.0);

                int samples = 0;
                for (int i = 0; i < 16; i++) {
                        // Sample random direction
                        vec3 wi = ggx_sample(material, n, wo, seed);

                        // Compute the BRDF and PDF
                        vec3 brdf = ggx_brdf(material, n, wo, wi) + material.diffuse/PI;
                        float pdf = ggx_pdf(material, n, wo, wi);

                        if (pdf < 1e-5f)
                                continue;

                        // Compute final radiance
                        color += brdf * env_radiance(wi).xyz * max(0.0, dot(wi, n))/pdf;
                        samples++;
                }

                radiance = vec4(color/samples, 1);
        } else {
                radiance = env_radiance(direction);
        }

        // vec4 radiance = vec4(direction, 1);
        imageStore(framebuffer, coord, radiance);
}
