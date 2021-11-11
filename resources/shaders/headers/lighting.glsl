struct PointLight {
	vec3 position;
	vec3 color;
};

vec3 point_light_contr(PointLight light, vec3 color, vec3 frag_pos, vec3 view_pos, vec3 normal)
{
	// ambient
	float ambient_strength = 0.1;
	vec3 ambient = ambient_strength * light.color;

	// diffuse
	vec3 norm = normalize(normal);
	if (!gl_FrontFacing)
		norm *= -1;

	vec3 light_dir = normalize(light.position - frag_pos);
	float diff = max(dot(norm, light_dir), 0.0);
	vec3 diffuse = diff * light.color;

	// specular
	float shine = 32;
	float specular_strength = 0.5;
	vec3 view_dir = normalize(view_pos - frag_pos);
	vec3 reflect_dir = reflect(-light_dir, norm);
	float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shine);
	vec3 specular = specular_strength * spec * light.color;

	vec3 result = (ambient + diffuse + specular) * color;

	// vec3 rgb_normal = 0.5 * norm + 0.5;
	return result;
}
