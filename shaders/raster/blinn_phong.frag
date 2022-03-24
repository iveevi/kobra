#version 450

#include "io_set.frag"

void main()
{
	vec3 light_pos = vec3(0.0, 0.0, 1.0);
	vec3 light_dir = normalize(vec3(0.0, 0.0, 1.0));

	vec3 n = normalize(normal);
	vec3 light_dir_normal = normalize(light_dir);

	vec3 light_color = vec3(1.0, 1.0, 1.0);
	vec3 ambient_color = vec3(0.2, 0.2, 0.2);
	vec3 diffuse_color = vec3(0.8, 0.8, 0.8);

	vec3 ambient = ambient_color * light_color;
	vec3 diffuse = diffuse_color * light_color * max(dot(n, light_dir_normal), 0.0);
	vec3 specular = vec3(0.0, 0.0, 0.0);

	vec3 c = ambient + diffuse + specular;

	// fragment = vec4(c, 1.0) * color;
	fragment = vec4(normal, 1.0);
}
