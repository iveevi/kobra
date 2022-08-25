#version 450

#include "bindings.h"
#include "io_set.glsl"

void main()
{
	fragment = vec4(mat.diffuse, 1.0);
}
