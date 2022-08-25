#version 450

#include "material.glsl"

// Only input is position
layout (location = 0) in vec3 position;

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

// Output direction from center
layout (location = 0) out vec3 direction;

void main()
{
	direction = position;
	gl_Position = proj * view * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position = gl_Position.xyww;
}
