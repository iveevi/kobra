#version 450

// Push constants for vertex shaders
// TODO: allow more flexible push constants
layout (push_constant) uniform PushConstants
{
	// Time
	float time;

	// Transform matrices
	mat4 model;
	mat4 view;
	mat4 proj;

	// Camera position
	vec3 view_pos;

	// Highlight the object?
	// TODO: remove
	float highlight;
};

// Only input is position
layout (location = 0) in vec3 position;

// Output direction from center
layout (location = 0) out vec3 direction;

void main()
{
	vec3 pos = mat3(view) * position.xyz; 
	gl_Position = (proj * vec4(pos, 0.0)).xyzz;
	gl_Position.y = -gl_Position.y;
	direction = position;
}
