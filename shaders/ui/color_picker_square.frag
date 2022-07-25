#version 450

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;

layout(location = 0) out vec4 fragment;

// Main function
void main()
{
	vec3 color = mix(vec3(1), in_color, in_uv.x) * (1 - in_uv.y);
	fragment = vec4(color, 1);
}
