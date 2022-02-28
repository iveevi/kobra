#version 450

// Vertex inputs
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

// Outputs
layout (location = 0) out vec3 fcolor;
layout (location = 1) out vec2 fpos;

// Main function
void main()
{
	// Set the vertex position
	gl_Position = vec4(position, 0.0, 1.0);
	
	// Fragment shader outputs
	fcolor = color;
	fpos = position;
}