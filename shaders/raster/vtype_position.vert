#version 450

// Typical vertex shader
layout(location = 0) in vec3 position;

// MVP matrix as push constant
layout (push_constant) uniform PushConstants
{
	mat4 model;
	mat4 view;
	mat4 projection;
};

// Out color
layout(location = 0) out vec4 color;

void main()
{
	// Transform vertex position by model, view and projection matrices
	gl_Position = projection * view * model * vec4(position, 1.0);

	// Output color
	color = vec4(1.0, 0.0, 0.0, 1.0);
}
