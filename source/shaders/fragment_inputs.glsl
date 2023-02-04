#include "bindings.h"
#include "material.glsl"

// Inputs
layout (location = 0) in vec3		position;
layout (location = 1) in vec3		normal;
layout (location = 2) in vec2		uv;
layout (location = 3) in mat3		tbn;
layout (location = 6) in vec3		tbn_inverse;
layout (location = 9) in vec3		camera_position;
layout (location = 10) flat in float	time;

// Uniform buffer for material
layout (binding = RASTER_BINDING_MATERIAL) uniform MaterialBlock
{
	Material mat;
};

// Sampler inputs
layout (binding = RASTER_BINDING_ALBEDO_MAP)
uniform sampler2D albedo_texture;

layout (binding = RASTER_BINDING_NORMAL_MAP)
uniform sampler2D normal_texture;

// Outputs
layout (location = 0) out vec4 fragment;
