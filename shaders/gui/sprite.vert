#version 450

// Vertex inputs
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 in_tex_coord;

// Outputs
layout (location = 0) out vec2 out_tex_coord;

// Main function
void main()
{
	// Set the vertex position
	gl_Position = vec4(position, 0.0, 1.0);

	// Set the texture coordinate
	out_tex_coord = in_tex_coord;
}
