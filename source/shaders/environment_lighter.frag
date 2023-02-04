#version 450

#include "fragment_inputs.glsl"

#define MIPS 5

layout (binding = 5) uniform sampler2D irradiance_maps[MIPS];

const float M_PI = 3.1415926535897932384626433832795;

void main()
{
	// TODO: fix this
	if (mat.has_albedo_texture > 0.5) {
		vec3 color = texture(albedo_texture, uv).rgb;
		fragment = vec4(color, 1);
	} else {
		fragment = vec4(mat.diffuse, 1.0);
	}

	vec3 view_dir = normalize(position - camera_position);

	// Convert direction to UV
	vec3 L = reflect(view_dir, normal);
	
	vec2 env_uv = vec2(0.0);
	env_uv.x = atan(L.x, L.z) / (2.0 * M_PI) + 0.5;
	env_uv.y = asin(L.y) / M_PI + 0.5;

	int mip = int(clamp(MIPS * mat.roughness, 0, MIPS - 1));
	fragment *= texture(irradiance_maps[mip], env_uv);
}
