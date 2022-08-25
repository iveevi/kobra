#version 450

#include "bindings.h"
#include "io_set.glsl"
#include "highlight.glsl"

void main()
{
	if (mat.has_albedo > 0.5) {
		vec3 color = texture(albedo_map, tex_coord).rgb;
		fragment = vec4(color, 1);
	} else {
		fragment = vec4(mat.diffuse, 1.0);
	}

	HL_OUT();
}
