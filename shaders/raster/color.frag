#version 450

// Input is color, output is color
layout (location = 0) in vec4 color;

layout (location = 0) out vec4 fragment;

void main()
{
	fragment = color;
}
