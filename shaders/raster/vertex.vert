#version 450

#include "material.glsl"

// Typical vertex shader
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

// MVP matrix as push constant
layout (push_constant) uniform PushConstants
{
	// Transform matrices
	mat4 model;
	mat4 view;
	mat4 proj;

	// Material properties
	Material material;
};

// Out variables
layout (location = 0) out vec3		out_position;
layout (location = 1) out vec3		out_normal;
layout (location = 2) out vec2		out_tex_coord;
layout (location = 3) out mat3		out_tbn;
layout (location = 6) out mat3		out_tbn_transpose;
layout (location = 9) out Material	out_material;

void main()
{
	// Transform vertex position by model, view and projection matrices
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;

	// TBN things
	vec3 vert_normal = normalize(normal);
	vec3 vert_tangent = normalize(tangent);
	vec3 vert_bitangent = normalize(bitangent);

	// Model view matrix
	mat3 mv_matrix = mat3(model);
	vert_normal = normalize(mv_matrix * vert_normal);
	vert_tangent = normalize(mv_matrix * vert_tangent);
	vert_bitangent = normalize(mv_matrix * vert_bitangent);
	mat3 tbn = mat3(vert_tangent, vert_bitangent, vert_normal);

	// Output necessary info
	out_position		= (model * vec4(position, 1.0)).xyz;
	out_normal		= vert_normal;
	out_tex_coord		= tex_coord;
	out_tbn			= tbn;
	out_tbn_transpose	= transpose(tbn);
	out_material		= material;
}
