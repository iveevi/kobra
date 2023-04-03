#version 450

#include "push_constants.glsl"

// Only input is position
layout (location = 0) in vec3 position;

// Output direction from center
layout (location = 0) out vec3 direction;

void main()
{
	direction = position;
	gl_Position = proj * view * vec4(position, 1.0);
	gl_Position = gl_Position.xyww;
}
