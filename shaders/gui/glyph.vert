#version 450

// Vertex inputs
layout(location = 0) in vec4 bounds;

// Outputs
layout (location = 0) out vec3 fcolor;
layout (location = 1) out vec2 fpos;

// Normalized vertex positions
vec2 nvecs[6] = vec2[6] (
	// Triangle 1
	vec2(0.0, 1.0),
	vec2(1.0, 1.0),
	vec2(0.0, 0.0),

	// Triangle 2
	vec2(1.0, 1.0),
	vec2(1.0, 0.0),
	vec2(0.0, 0.0)
);

// Main function
void main()
{
	// Create positions
	vec2 pos[6] = vec2[6] (
		// Triangle 1
		vec2(bounds.x, bounds.y),
		vec2(bounds.z, bounds.y),
		vec2(bounds.x, bounds.w),

		// Triangle 2
		vec2(bounds.z, bounds.y),
		vec2(bounds.z, bounds.w),
		vec2(bounds.x, bounds.w)
	);

	// Set the vertex position
	gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
	
	// Fragment shader outputs
	fcolor = vec3(1.0, 1.0, 1.0);
	fpos = nvecs[gl_VertexIndex];
}
