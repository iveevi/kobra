#version 450

// Inputs
layout (location = 0) in vec2 in_tex_coord;

layout (binding = 0) uniform sampler2D sprite;

// Outputs
layout (location = 0) out vec4 fragment;

// Main function
void main()
{
	fragment = texture(sprite, in_tex_coord);
}
