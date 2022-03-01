#version 450

// Vertex inputs
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

// Outputs
layout (location = 0) out vec3 fcolor;
layout (location = 1) out vec2 fpos;

// Normalized vertex positions
vec2 nvecs[4] = vec2[4](
	vec2(0.0, 0.0),
	vec2(1.0, 0.0),
	vec2(1.0, 1.0),
	vec2(0.0, 1.0)
);


// Main function
void main()
{
	// Set the vertex position
	gl_Position = vec4(position, 0.0, 1.0);
	
	// Fragment shader outputs
	fcolor = color;
	fpos = nvecs[gl_VertexIndex];
}
