// Material structure
struct Material {
	vec3 albedo;
	float shading;

	float specular;
	float reflectance;
	float refractance;
};

// Default "constructor"
Material mat_default()
{
	return Material(
		vec3(0.5f, 0.5f, 0.5f), 0.0f,
		0.0f, 0.0f, 0.0f
	);
}
