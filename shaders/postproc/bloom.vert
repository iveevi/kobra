#version 430

// NDC full square vertices
vec2 ndc_vertices[6] = vec2[6] (
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0,  1.0), vec2(-1.0, -1.0), vec2( 1.0,  1.0)
);

void main()
{
	gl_Position = vec4(ndc_vertices[gl_VertexIndex], 0.0, 1.0);
}