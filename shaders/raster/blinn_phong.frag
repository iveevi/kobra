#version 450

// Modules
#include "io_set.glsl"
#include "light_set.glsl"
#include "highlight.glsl"

// TODO: move to light_set module
vec3 point_light(vec3 light_position, vec3 position)
{
	// Blinn Phong
	vec3 light_dir = normalize(light_position - position);
	vec3 normal = normalize(normal);

	float intensity = 5/distance(light_position, position);

	float diffuse = max(dot(light_dir, normal), 0.0);
	float specular = pow(
		max(
			dot(light_dir, reflect(-light_dir, normal)),
			0.0
		), 32
	);

	return material.albedo * intensity * (diffuse);
}

void main()
{
	// First check if the object is emissive
	if (abs(material.shading_type - 5.0) < 0.01) {
		fragment = vec4(material.albedo, 1.0);
		HL_OUT();
		return;
	}

	// Sum up the light contributions
	vec3 color = vec3(0.0);

	for (int i = 0; i < point_lights.number; i++)
		color += point_light(point_lights.positions[i], position);

	// Gamma correction
	color = pow(color, vec3(1.0 / 2.2));

	// Output
	fragment = vec4(color, 1.0);
	HL_OUT();
}
