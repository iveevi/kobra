#version 450

// Modules
#include "bindings.h"
#include "io_set.glsl"
#include "highlight.glsl"

void main()
{
	vec3 n = normalize(normal);
	if (mat.has_normal > 0.5) {
		n = texture(normal_map, tex_coord).rgb;
		n = 2.0 * n - 1.0;
		n = normalize(tbn * n);
	}

	fragment = vec4(n * 0.5f + 0.5f, 1.0);
	HL_OUT();
}
