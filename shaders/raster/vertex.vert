#version 450

// Typical vertex shader
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

// MVP matrix as push constant
layout (push_constant) uniform PushConstants
{
	mat4 model;
	mat4 view;
	mat4 projection;
};

// Out color
layout (location = 0) out vec4 color;
layout (location = 1) out vec3 normal_out;
layout (location = 2) out vec2 tex_coord_out;

void main()
{
	// Transform vertex position by model, view and projection matrices
	gl_Position = projection * view * model * vec4(position, 1.0);

	// Output necessary info
	color = vec4(1.0, 0.0, 0.0, 1.0);
	normal_out = normal;
	tex_coord_out = tex_coord;
}
