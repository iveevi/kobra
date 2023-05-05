#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)
uniform writeonly image2D framebuffer;

layout (binding = 1)
uniform sampler2D environment;

layout (push_constant) uniform PushConstants {
        vec3 origin;
        vec3 forward;
        vec3 up;
        vec3 right;
} pushConstants;

// Constants
float PI = 3.14159265358979323846;

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

        vec2 uv = (vec2(coord) + vec2(0.5)) / vec2(size);
        uv = 2.0 * uv - 1.0;

        // Compute the ray direction
        vec3 direction = normalize(pushConstants.forward
                + uv.x * pushConstants.right
                + uv.y * pushConstants.up);

        // Compute UV on the environment map
        vec3 x;
        vec3 n;

        vec4 radiance;
        if (hit_sphere(pushConstants.origin, direction, x, n)) {
                vec3 normal = normalize(x - vec3(0, 0, 0));
                vec3 color = 0.5 * (normal + vec3(1.0));
                radiance = vec4(color, 1);

                vec3 direction = reflect(direction, normal);
                radiance *= env_radiance(direction);
        } else {
                radiance = env_radiance(direction);
        }

        // vec4 radiance = vec4(direction, 1);
        imageStore(framebuffer, coord, radiance);
}
