#version 450

// Vertex inputs
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 color;

// Outputs
layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out vec2 out_position;

// Main function
void main()
{
	// Set the vertex position
	gl_Position = vec4(position, 0.0, 1.0);
	
	// Set the outputs
	out_color = color;
	out_uv = uv;
	out_position = position;
}
