#version 450

#include "fragment_inputs.glsl"

#define MIPS 5

layout (binding = 5) uniform sampler2D irradiance_maps[MIPS];

const float M_PI = 3.1415926535897932384626433832795;

void main()
{
	// Load material and surface properties
	vec3 diffuse = mat.diffuse;
	vec3 specular = mat.specular;
	vec3 n = normal;

	if (mat.has_albedo_texture > 0.5) {
		vec4 full_diffuse = texture(albedo_texture, uv);
		if (full_diffuse.a < 0.5)
			discard;
		diffuse = full_diffuse.rgb;
	}

	if (mat.has_normal_texture > 0.5) {
		n = texture(normal_texture, uv).rgb;
		n = normalize(tbn * (2 * n - 1));
	}

	// Compute reflection vector
	vec3 view_dir = normalize(position - camera_position);
	vec3 L = reflect(view_dir, normal);

	// Convert direction to UV
	vec2 env_uv = vec2(0.0);
	env_uv.x = atan(L.x, L.z) / (2.0 * M_PI) + 0.5;
	env_uv.y = asin(L.y) / M_PI + 0.5;

	// Compute final color
	diffuse *= texture(irradiance_maps[MIPS - 1], env_uv).rgb;

	int mip_low = int(floor(MIPS * mat.roughness));
	int mip_high = int(clamp(ceil(MIPS * mat.roughness), 0, MIPS - 1));

	vec3 irr_low = texture(irradiance_maps[mip_low], env_uv).rgb;
	vec3 irr_high = texture(irradiance_maps[mip_high], env_uv).rgb;

	specular *= mix(irr_low, irr_high, MIPS * mat.roughness - float(mip_low));
	fragment = vec4(diffuse + specular, 1.0);
}
