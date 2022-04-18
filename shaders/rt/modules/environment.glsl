// Sample environment map
vec3 sample_environment(Ray ray)
{
	// Get uv coordinates
	vec2 uv = vec2(0.0);
	uv.x = atan(ray.direction.x, ray.direction.z) / (2.0 * PI) + 0.5;
	uv.y = asin(ray.direction.y) / PI + 0.5;

	// Get the color
	vec3 tex = texture(s2_environment, uv).rgb;

	return tex;
}

// Sample environment map wih blur
vec3 sample_environment_blur(Ray ray)
{
	int samples = 16;

	vec3 color = vec3(0.0);
	for (int i = 0; i < samples; i++) {
		vec2 uv = vec2(0.0);
		uv.x = atan(ray.direction.x, ray.direction.z) / (2.0 * PI) + 0.5;
		uv.y = asin(ray.direction.y) / PI + 0.5;

		vec2 j = jitter2d(samples, i);
		vec3 tex = texture(s2_environment, uv + 0.025 * j).rgb;
		color += tex;
	}

	return color / float(samples);
}
