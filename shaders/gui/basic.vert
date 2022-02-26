#version 450

// Vertex inputs
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

// Outputs
layout (location = 0) out vec3 fcolor;

// Main function
void main()
{
	// Set the vertex position
	gl_Position = vec4(position, 0.0, 1.0);
	
	// Set the color
	fcolor = color;
}