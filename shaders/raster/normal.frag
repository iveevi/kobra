#version 450

#include "io_set.glsl"
#include "highlight.glsl"

void main()
{
	fragment = vec4(normalize(normal * 0.5f) + 0.5f, 1.0); 
	HL_OUT();
}
