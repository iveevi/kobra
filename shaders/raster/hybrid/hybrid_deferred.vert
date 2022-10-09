#version 450

// Typical vertex shader
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec3 in_tangent;
layout (location = 4) in vec3 in_bitangent;

// Push constants
layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
	int index;
};

// G-buffer outputs
layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out mat3 out_tbn;
layout (location = 6) out int out_id;

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
	out_uv = in_uv;
	out_tbn = tbn;
	out_id = index;
}
