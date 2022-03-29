#version 450

#include "io_set.glsl"

vec3 light = vec3(0.0, 3.0, 0.0);

void main()
{
	// Blinn Phong
	vec3 albedo = material.albedo;

	vec3 light_dir = normalize(light - position);
	vec3 normal = normalize(normal);

	float diffuse = max(dot(light_dir, normal), 0.0);
	float specular = pow(
		max(
			dot(light_dir, reflect(-light_dir, normal)),
			0.0
		), 32
	);

	vec3 color = albedo * (diffuse + specular);

	// Output
	fragment = vec4(color, 1.0);
}
