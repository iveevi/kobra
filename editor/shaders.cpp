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
        int texture_status;
};

// G-buffer outputs
// TODO: remove the extra outputs
layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out mat3 out_tbn;
layout (location = 6) out int out_material_index;
layout (location = 7) out int out_texture_status;

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
        out_texture_status = texture_status;
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
layout (location = 7) flat in int in_texture_status;

// Sampler inputs
layout (binding = 0)
uniform sampler2D albedo_map;

layout (binding = 1)
uniform sampler2D normal_map;

// TODO: list of objects to highlight

// Outputs
layout (location = 0) out vec4 g_position;
layout (location = 1) out vec4 g_normal;
layout (location = 2) out int g_material_index;

void main()
{
	g_position = vec4(in_position, 1.0);

        // If albedo map is present then mask out fragments
        int has_albedo = in_texture_status & 0x1;
        if (has_albedo != 0) {
                if (texture(albedo_map, in_uv).a < 0.5)
                        discard;
        }

        // Normal map if present
        int has_normal = in_texture_status & 0x2;
	if (has_normal != 0) {
		g_normal.xyz = 2 * texture(normal_map, in_uv).rgb - 1;
		g_normal.xyz = normalize(in_tbn * g_normal.xyz);
	} else {
                g_normal.xyz = normalize(in_normal);
        }

        g_normal.w = 1.0;

        // Set material index
	g_material_index.x = in_material_index | (gl_PrimitiveID << 16);
}
)";

// TODO: use a simple mesh shader for this...
const char *presentation_vert_shader = R"(
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

const char *normal_frag_shader = R"(
#version 450

layout (binding = 0)
uniform sampler2D normal_map;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 fragment;

void main()
{
        vec3 normal = texture(normal_map, in_uv).xyz;
        fragment = vec4(0.5 * normal + 0.5, 1.0);
}
)";

const char *triangulation_frag_shader = R"(
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
        if (index == -1)
                discard;

        int triangle_index = index >> 16;
        vec3 color = COLOR_WHEEL[triangle_index % 12];
        fragment = vec4(color, 1.0);

        // Mix with the outline
        fragment = mix(fragment, vec4(0, 0, 0, 1), sobel);
}
)";

// Fragment shader for highlighting (outline)
const char *highlight_frag_shader = R"(
#version 450

layout (binding = 0)
uniform isampler2D index_map;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 fragment;

// Push constants
layout (push_constant) uniform PushConstants {
        vec4 color;
        int index;
} push_constants;

void main()
{
        int index = texture(index_map, in_uv).x;
        index = index & 0xFFFF;

        if (index == push_constants.index)
                fragment = push_constants.color;
        else
                discard;
}
)";

// Albedo rendering
const char *albedo_vert_shader = R"(
#version 450

layout (location = 0) in vec3 in_position;
layout (location = 2) in vec2 in_uv;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
        vec4 color;
        int has_albedo;
};

layout (location = 0) out vec2 out_uv;
layout (location = 1) out vec4 out_color;
layout (location = 2) out int out_has_albedo;

void main()
{
	// First compute rendering position
	gl_Position = proj * view * model * vec4(in_position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w)/2.0;
	
	// Pass outputs
	out_uv = vec2(in_uv.x, 1.0 - in_uv.y);
        out_color = color;
        out_has_albedo = has_albedo;
}
)";

const char *albedo_frag_shader = R"(
#version 450

layout (binding = 0)
uniform sampler2D albedo_map;

layout (location = 0) in vec2 in_uv;
layout (location = 1) in vec4 in_color;
layout (location = 2) flat in int in_has_albedo;

layout (location = 0) out vec4 fragment;

void main()
{
        if (in_has_albedo != 0) {
                fragment = texture(albedo_map, in_uv);
                if (fragment.a < 0.5)
                        discard;
        } else {
                fragment = in_color;
        }
}
)";

// Bounding box shaders
const char *bounding_box_vert_shader = R"(
#version 450

layout (location = 0) in vec3 in_position;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
};

void main()
{
	// First compute rendering position
	gl_Position = proj * view * model * vec4(in_position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w)/2.0;
}
)";

const char *bounding_box_frag_shader = R"(
#version 450

layout (push_constant) uniform PushConstants {
        layout (offset = 192) vec4 color;
};

layout (location = 0) out vec4 fragment;

void main()
{
        fragment = color;
}
)";

// Compute shader to apply Sobel filter to the normal map
const char *sobel_comp_shader = R"(
#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, r32i)
uniform readonly iimage2D index_map;

layout (binding = 1, r32f)
uniform writeonly image2D sobel_map;

int sobel_kernel_x[3][3] = int[3][3](
        int[3](1, 0, -1),
        int[3](2, 0, -2),
        int[3](1, 0, -1)
);

int sobel_kernel_y[3][3] = int[3][3](
        int[3](1, 2, 1),
        int[3](0, 0, 0),
        int[3](-1, -2, -1)
);

float sobel_kernel(ivec2 coord)
{
        int sobel_x = 0;
        int sobel_y = 0;

        for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                        ivec2 offset = ivec2(x, y);

                        int id = imageLoad(index_map, coord + offset).r;
                        id = id & 0xFFFF;

                        sobel_x += sobel_kernel_x[y + 1][x + 1] * id;
                        sobel_y += sobel_kernel_y[y + 1][x + 1] * id;
                }
        }



        return min(abs(sobel_x) + abs(sobel_y), 1.0);
}

void main()
{
        ivec2 size = imageSize(index_map);
        ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

        // Make sure we don't go out of bounds
        bool x_in_bounds = coord.x >= 1 && coord.x < size.x - 1;
        bool y_in_bounds = coord.y >= 1 && coord.y < size.y - 1;

        if (!x_in_bounds || !y_in_bounds) {
                imageStore(sobel_map, coord, ivec4(0));
                return;
        }

        float sobel = sobel_kernel(coord);
        imageStore(sobel_map, coord, vec4(sobel));

        // If possible, perform a blur on the sobel map
        x_in_bounds = coord.x >= 2 && coord.x < size.x - 2;
        y_in_bounds = coord.y >= 2 && coord.y < size.y - 2;

        if (!x_in_bounds || !y_in_bounds)
                return;

        // Compute the average sobel value around the neighborhood
        for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                        ivec2 offset = ivec2(x, y);
                        sobel += sobel_kernel(coord + offset);
                }
        }

        sobel /= 9.0;
        imageStore(sobel_map, coord, vec4(sobel));
}
)";
