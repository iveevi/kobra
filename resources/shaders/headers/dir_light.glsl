struct DirLight {
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

vec3 dir_light_contr(DirLight light, vec3 color, vec3 frag_pos, vec3 view_pos, vec3 normal)
{
        // Ambient
	vec3 ambient = light.ambient;

	// Diffuse
	vec3 norm = normalize(normal);
	if (!gl_FrontFacing)
		norm *= -1;

	vec3 light_dir = normalize(-light.direction);
	float diff = max(dot(norm, light_dir), 0.0);
	vec3 diffuse = diff * light.diffuse;

	// Specular
	float shine = 32;

	vec3 view_dir = normalize(view_pos - frag_pos);
	vec3 refl_dir = reflect(-light_dir, norm);
	float spec = pow(max(dot(view_dir, refl_dir), 0.0), shine);
	vec3 specular = spec * light.specular;

	// Total result
	vec3 result = (ambient + diffuse + specular) * color;
	return result;
}