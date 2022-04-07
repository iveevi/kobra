#version 450

#include "io_set.glsl"
#include "highlight.glsl"

void main()
{
	fragment = vec4(material.albedo, 1.0);
	HL_OUT();
}
