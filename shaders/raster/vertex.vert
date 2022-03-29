#version 450

#include "material.glsl"

// Typical vertex shader
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

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

// Out color
layout (location = 0) out vec3		out_position;
layout (location = 1) out vec3		out_normal;
layout (location = 2) out vec2		out_tex_coord;
layout (location = 3) out Material	out_material;

void main()
{
	// Transform vertex position by model, view and projection matrices
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;

	// Output necessary info
	out_position	= (model * vec4(position, 1.0)).xyz;
	out_normal	= normal;
	out_tex_coord	= tex_coord;
	out_material	= material;
}
