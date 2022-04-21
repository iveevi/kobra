#version 450

// Import bindings
#include "../bindings.h"

// Input texture
layout (set = 0, binding = MESH_BINDING_PIXELS)
uniform sampler2D pixels;

// Input is texture coordinate
layout (location = 0) in vec2 coord;

// Output color
layout (location = 0) out vec4 fragment;

void main()
{
	fragment = texture(pixels, coord);
}
