const char *gbuffer_vert_shader = R"(
#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec3 in_tangent;
layout (location = 4) in vec3 in_bitangent;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
	int material_index;
};

// G-buffer outputs
layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out mat3 out_tbn;
layout (location = 6) out int out_material_index;

void main()
{
	// First compute rendering position
	gl_Position = proj * view * model * vec4(in_position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w)/2.0;
	
	// TBN things
	vec3 vert_normal = normalize(in_normal);
	vec3 vert_tangent = normalize(in_tangent);
	vec3 vert_bitangent = normalize(in_bitangent);
	
	// Model matrix
	mat3 mv_matrix = mat3(model);
	
	vert_normal = normalize(mv_matrix * vert_normal);
	vert_tangent = normalize(mv_matrix * vert_tangent);
	vert_bitangent = normalize(mv_matrix * vert_bitangent);

	mat3 tbn = mat3(vert_tangent, vert_bitangent, vert_normal);

	// Pass outputs
	out_position = vec3(model * vec4(in_position, 1.0));
	out_normal = normalize(mv_matrix * in_normal);
	out_uv = vec2(in_uv.x, 1.0 - in_uv.y);
	out_tbn = tbn;
        out_material_index = material_index;
}
)";

const char *gbuffer_frag_shader = R"(
#version 450

// Inputs
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in mat3 in_tbn;
layout (location = 6) flat in int in_material_index;

// Sampler inputs
layout (binding = 0)
uniform sampler2D albedo_map;

layout (binding = 1)
uniform sampler2D normal_map;

// Outputs
layout (location = 0) out vec4 g_position;
layout (location = 1) out vec4 g_normal;
layout (location = 2) out int g_material_index;

void main()
{
        // TODO: if albedo map is valid then mask out fragments...
	g_position = vec4(in_position, 1.0);
	g_normal = vec4(in_normal, 1.0);

        // TODO: pass pushconstant to vertex shader if normal map is present

	/* Normal (TODO: use int instead of float for has_normal)
	if (mat.has_normal > 0.5) {
		g_normal.xyz = 2 * texture(normal_map, in_uv).rgb - 1;
		g_normal.xyz = normalize(in_tbn * g_normal.xyz);
	} else {
		g_normal = vec4(in_normal, 1.0);
	} */

	g_material_index = in_material_index | (gl_PrimitiveID << 16);
}
)";

// TODO: use a simple mesh shader for this...
const char *present_vert_shader = R"(
#version 450

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_uv;

layout (location = 0) out vec2 out_uv;

void main()
{
        gl_Position = vec4(in_position, 0.0, 1.0);
        out_uv = in_uv;
}
)";

const char *present_frag_shader = R"(
#version 450

layout (binding = 0)
uniform sampler2D position_map;

layout (binding = 1)
uniform sampler2D normal_map;

layout (binding = 2)
uniform isampler2D index_map;

layout (binding = 3)
uniform sampler2D sobel_map;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 fragment;

// Color wheel with differing hues (HSL)
vec3 COLOR_WHEEL[12] = vec3[12](
        vec3(0.910, 0.490, 0.490),
        vec3(0.910, 0.700, 0.490),
        vec3(0.910, 0.910, 0.490),
        vec3(0.700, 0.910, 0.490),
        vec3(0.490, 0.910, 0.490),
        vec3(0.490, 0.910, 0.700),
        vec3(0.490, 0.910, 0.910),
        vec3(0.490, 0.700, 0.910),
        vec3(0.490, 0.490, 0.910),
        vec3(0.700, 0.490, 0.910),
        vec3(0.910, 0.490, 0.910),
        vec3(0.910, 0.490, 0.700)
);

void main()
{
        vec3 position = texture(position_map, in_uv).xyz;
        vec3 normal = texture(normal_map, in_uv).xyz;
        float sobel = texture(sobel_map, in_uv).r;
        int index = int(texture(index_map, in_uv).r);

        int material_index = index & 0xFFFF;
        if (index == -1 || sobel > 1)
                discard;

        // fragment = vec4(0.5 * normal + 0.5, 1.0);
        int triangle_index = index >> 16;
        vec3 color = COLOR_WHEEL[triangle_index % 12];
        fragment = vec4(color, 1.0);
}
)";

// Compute shader to apply Sobel filter to the normal map
const char *sobel_comp_shader = R"(
#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)
uniform readonly image2D normal_map;

layout (binding = 1, r32f)
uniform writeonly image2D sobel_map;

float sobel_kernel_x[3][3] = float[3][3](
        float[3](1.0, 0.0, -1.0),
        float[3](2.0, 0.0, -2.0),
        float[3](1.0, 0.0, -1.0)
);

float sobel_kernel_y[3][3] = float[3][3](
        float[3](1.0, 2.0, 1.0),
        float[3](0.0, 0.0, 0.0),
        float[3](-1.0, -2.0, -1.0)
);

void main()
{
        ivec2 size = imageSize(normal_map);
        ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

        // Make sure we don't go out of bounds
        bool x_in_bounds = coord.x >= 1 && coord.x < size.x - 1;
        bool y_in_bounds = coord.y >= 1 && coord.y < size.y - 1;

        if (!x_in_bounds || !y_in_bounds) {
                imageStore(sobel_map, coord, vec4(0));
                return;
        }

        // if (coord.x >= size.x || coord.y >= size.y)
        //        return;

        vec3 normal = imageLoad(normal_map, coord).xyz;
        vec3 sobel_x = vec3(0.0);
        vec3 sobel_y = vec3(0.0);

        for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                        ivec2 offset = ivec2(x, y);
                        vec3 n = imageLoad(normal_map, coord + offset).xyz;
                        sobel_x += sobel_kernel_x[y + 1][x + 1] * n;
                        sobel_y += sobel_kernel_y[y + 1][x + 1] * n;
                }
        }

        float sobel_magnitude = length(sobel_x) + length(sobel_y);

        imageStore(sobel_map, coord, vec4(sobel_magnitude));
        // imageStore(sobel_map, coord, vec4(sobel, 1.0));
}
)";
