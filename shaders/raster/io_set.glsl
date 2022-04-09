#include "material.glsl"

// Inputs
layout (location = 0) in vec3		position;
layout (location = 1) in vec3		normal;
layout (location = 2) in vec2		tex_coord;
layout (location = 3) in mat3		tbn;
layout (location = 6) in vec3		tbn_inverse;
layout (location = 9) in Material	material;

// Sampler inputs
layout (binding = RASTER_BINDING_ALBEDO_MAP)
uniform sampler2D			albedo_map;

layout (binding = RASTER_BINDING_NORMAL_MAP)
uniform sampler2D			normal_map;

// Outputs
layout (location = 0) out vec4		fragment;
