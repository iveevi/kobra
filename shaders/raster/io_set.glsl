#include "material.glsl"

// Inputs
layout (location = 0) in vec3		position;
layout (location = 1) in vec3		normal;
layout (location = 2) in vec2		tex_coord;
layout (location = 3) in Material	material;

// Outputs
layout (location = 0) out vec4		fragment;
