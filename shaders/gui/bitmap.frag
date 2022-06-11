#version 450

// Inputs
layout (location = 0) in vec3 fcolor;
layout (location = 1) in vec2 fpos;

layout (binding = 0) uniform sampler2D tex;

// Outputs
layout (location = 0) out vec4 fragment;

// Main function
void main()
{
	float t = texture(tex, fpos).r;
	fragment = vec4(fcolor, 1.0) * t;
}
