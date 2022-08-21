#version 450

// Modules
#include "bindings.h"
#include "io_set.glsl"
#include "light_set.glsl"
#include "highlight.glsl"

const float PI = 3.1415926535897932384626433832795;

void main()
{
	// TODO: Kd anbd Ks interface
	vec3 albedo = material.albedo;
	vec3 n = normalize(normal);
	
	if (material.has_albedo > 0.5)
		albedo = texture(albedo_map, tex_coord).rgb;

	if (material.has_normal > 0.5) {
		n = texture(normal_map, tex_coord).rgb;
		n = 2 * n - 1;
		n = normalize(tbn * n);
	}

	// First check if the object is emissive
	if (is_type(material.type, SHADING_EMISSIVE)) {
		fragment = vec4(albedo, 1.0);
		HL_OUT();
		return;
	}

	// Sum up the light contributions
	vec3 color = vec3(0.0);

	for (int i = 0; i < light_count; i++) {
		Light light = lights[i];

		// TODO: proper area light interface

		vec3 ldir = normalize(light.position - position);
		float d = distance(light.position, position);

		vec3 Li = light.intensity/(d * d);

		vec3 diffuse = albedo * max(dot(ldir, n), 0.0)/PI;
		float specular = pow(
			max(
				dot(ldir, reflect(-ldir, n)),
				0.0
			), 1 // TODO: shininess
		);

		// TODO: view position, material shininess, etc
		color += Li * (diffuse) + 0.05;
	}

	// Gamma correction
	color = pow(color, vec3(1.0 / 2.2));

	// Output
	fragment = vec4(color, 1.0);
	HL_OUT();
}
