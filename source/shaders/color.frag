#version 450

#include "fragment_inputs.glsl"

void main()
{
	// TODO: fix this
	if (mat.has_albedo_texture > 0.5) {
		vec3 color = texture(albedo_texture, uv).rgb;
		fragment = vec4(color, 1);
	} else {
		fragment = vec4(mat.diffuse, 1.0);
	}
}
