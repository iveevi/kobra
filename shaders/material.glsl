// Material structure
struct Material {
	vec3 albedo;
	float shading;

	float specular;
	float reflectance;
	
	// Index of refraction as a complex number
	vec2 ior;
};

// Default "constructor"
Material mat_default()
{
	return Material(
		vec3(0.5f, 0.5f, 0.5f), 0.0f,
		0.0f, 0.0f, vec2(0.0f, 0.0f)
	);
}
