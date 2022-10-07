#version 450

// NDC full square vertices
vec2 ndc_vertices[6] = vec2[6] (
    vec2( 1.0, -1.0), vec2(-1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2(-1.0,  1.0), vec2( 1.0,  1.0)
);

// NDC full square texture coordinates
vec2 ndc_texcoords[6] = vec2[6] (
    vec2(1.0, 0.0), vec2(0.0, 0.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0)
);

// Output texture coordinates
layout (location = 0) out vec2 coord;

void main()
{
	gl_Position = vec4(ndc_vertices[gl_VertexIndex], 0.0, 1.0);
	coord = ndc_texcoords[gl_VertexIndex];
}
