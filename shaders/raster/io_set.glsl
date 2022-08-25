#include "material.glsl"

// Inputs
layout (location = 0) in vec3		position;
layout (location = 1) in vec3		normal;
layout (location = 2) in vec2		tex_coord;
layout (location = 3) in mat3		tbn;
layout (location = 6) in vec3		tbn_inverse;
layout (location = 9) in vec3		view_pos;
layout (location = 10) flat in float	time;
layout (location = 11) flat in float	highlight;

// TODO: fix the location sparsity

// Uniform buffer for material
layout (binding = RASTER_BINDING_UBO) uniform MaterialBlock
{
	Material mat;
};

// Sampler inputs
layout (binding = RASTER_BINDING_ALBEDO_MAP)
uniform sampler2D			albedo_map;

layout (binding = RASTER_BINDING_NORMAL_MAP)
uniform sampler2D			normal_map;

// Outputs
layout (location = 0) out vec4		fragment;
