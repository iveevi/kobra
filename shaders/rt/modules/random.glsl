// Golden random function
float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio

float ray_seed = 1.0;

float gold_noise(in vec2 xy, in float seed)
{
	return fract(sin(dot(xy, vec2(12.9898, 78.233))) * seed);
}

// 2D Jittering
vec2 jitter2d(in vec2 xy, float strata, float i)
{
	float rx = gold_noise(vec2(xy.x, xy.y + i), ray_seed);
	float ry = gold_noise(vec2(xy.y + i, xy.x), ray_seed);

	// Get into the range [-0.5, 0.5]
	rx -= 0.5;
	ry -= 0.5;

	// Size of each square
	float inv = 1.0 / strata;
	float ix = floor(i/strata);
	float iy = i - ix * strata;

	// Center of the ith square
	float cx = ix * inv + 0.5;
	float cy = iy * inv + 0.5;

	// Jitter from the center of the ith square
	float x = rx * inv + cx;
	float y = ry * inv + cy;

	// Update seed
	ray_seed = fract((ray_seed + 1.0) * PHI);

	return vec2(x, y);
}
