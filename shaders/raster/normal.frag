#version 450

#include "io_set.glsl"

void main()
{
	fragment = vec4(normalize(normal * 0.5f) + 0.5f, 1.0); 
}
