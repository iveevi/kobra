// Constants
float PI = 3.14159265358979323846;
float PHI = 1.61803398874989484820;

// Random seed
vec3 random_seed = vec3(0.0, 0.0, 0.0);

// http://www.jcgt.org/published/0009/03/02/
uvec3 pcg3d(uvec3 v)
{
	v = v * 1664525u + 1013904223u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v ^= v >> 16u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	return v;
}

vec3 random3(vec3 f)
{
	return uintBitsToFloat((pcg3d(floatBitsToUint(f)) & 0x007FFFFFu) | 0x3F800000u) - 1.0;
}

float random(vec2 xy, float z) 
{
	return random3(vec3(xy, z)).x;
}

// 2D Jittering
vec2 jitter2d(float strata, float i)
{
	// Generate random numbers
	random_seed = random3(random_seed);

	// Get into the range [-0.5, 0.5]
	float rx = 0.5 * random_seed.x;
	float ry = 0.5 * random_seed.y;

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

	return vec2(x, y);
}

// Random point in unit sphere
vec3 random_sphere()
{
	// Generate random vec3
	random_seed = random3(random_seed);

	float ang1 = (random_seed.x + 1.0) * PI;
	float u = random_seed.y;
	float u2 = u * u;
	
	float sqrt1MinusU2 = sqrt(1.0 - u2);
	
	float x = sqrt1MinusU2 * cos(ang1);
	float y = sqrt1MinusU2 * sin(ang1);
	float z = u;

	return vec3(x, y, z);
}

// Random point in unit hemisphere
vec3 random_hemi(vec3 normal)
{
	vec3 v = random_sphere();
	return v * sign(dot(v, normal));
}
