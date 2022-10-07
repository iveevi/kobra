#version 450

// Input texture and coordinates
layout (binding = 0) uniform sampler2D pixels;

layout (location = 0) in vec2 coord;

// Output color
layout (location = 0) out vec4 fragment;

void main()
{
	fragment = texture(pixels, coord);
	fragment.a = 1.0;
}
