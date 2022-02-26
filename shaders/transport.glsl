// Light transport calculations
vec3 diffuse(Hit hit, Ray ray)
{
	// If no hit, just return background color
	if (hit.object < 0)
		return hit.mat.albedo;
	
	// Check for light type again
	if (hit.mat.shading == SHADING_TYPE_LIGHT)
		return hit.mat.albedo;

	// Intersection bias
	float bias = 0.1;

	// Light position (fixed for now)
	vec3 light_position = lights.data[0].yzw; // TODO: function to get light position
	vec3 light_direction = normalize(light_position - hit.point);

	// Shadow calculation
	vec3 shadow_origin = hit.point + hit.normal * bias;

	Ray shadow_ray = Ray(
		shadow_origin,
		light_direction
	);

	Hit shadow_hit = closest_object(shadow_ray);

	float shadow = 0.0;
	if (shadow_hit.object >= 0 && shadow_hit.mat.shading != SHADING_TYPE_LIGHT)
		shadow = 1.0;

	// Diffuse
	float diffuse = max(dot(hit.normal, light_direction), 0.0);

	// Combine into one factor
	float ambience = 0.15;
	float factor = ambience + (1.0 - ambience) * (diffuse);

	// Return final color
	return hit.mat.albedo * factor;
}