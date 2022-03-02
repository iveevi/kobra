#version 450

// Inputs
layout (location = 0) in vec3 fcolor;
layout (location = 1) in vec2 fpos;

layout (binding = 0) uniform sampler2D tex;

// Outputs
layout (location = 0) out vec4 color;

// Main function
void main()
{
	color = texture(tex, fpos);
}
