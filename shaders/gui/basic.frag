#version 450

// Inputs
layout(location = 0) in vec3 in_color;

// Outputs
layout(location = 0) out vec4 fragment;

// Main function
void main()
{
	fragment = vec4(in_color, 1.0);
}
