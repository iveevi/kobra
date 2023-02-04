// Material properties
struct Material {
	vec3 diffuse;
	vec3 specular;
	vec3 emission;
	vec3 ambient;

	float shininess;
	float roughness;
		
	int type;
	float has_albedo_texture; // TODO: encode into a single int
	float has_normal_texture;
};
