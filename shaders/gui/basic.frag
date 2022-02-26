#version 450

// Inputs
layout(location = 0) in vec3 fcolor;

// Outputs
layout(location = 0) out vec4 color;

// Main function
void main()
{
	color = vec4(fcolor, 1.0);
}