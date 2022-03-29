#version 450

#include "io_set.glsl"

void main()
{
	fragment = vec4(material.albedo, 1.0);
}
