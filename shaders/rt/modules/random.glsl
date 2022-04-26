// Constants
const float PI = 3.14159265358979323846;
const float INV_PI = 0.31830988618379067154;
const float PHI = 1.61803398874989484820;

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

float random() 
{
	random_seed = random3(random_seed);
	return fract(random_seed.x + random_seed.y + random_seed.z);
}

// 2D Jittering
vec2 jitter2d(float strata, float i)
{
	// Generate random numbers
	random_seed = random3(random_seed);

	// Get into the range [-0.5, 0.5]
	vec2 r = 0.5 * random_seed.xy;
	
	float inv_strata = 1.0/strata;
	float ix = floor(i * inv_strata);
	
	vec2 ir = vec2(ix, i - ix * strata);
	vec2 center = ir * inv_strata + 0.5;
	return r * inv_strata + center;
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

// Concentric disk sampling
vec2 concentric_disk()
{
	// Generate random numbers
	random_seed = random3(random_seed);

	// Get into the range [-1, 1]
	vec2 r = 2.0 * random_seed.xy - 1.0;

	// Degenerate case
	if (r.x == 0.0 && r.y == 0.0)
		return vec2(0.0, 0.0);

	// Apply concentric mapping
	float theta;
	float radius;

	if (abs(r.x) > abs(r.y)) {
		radius = r.x;
		theta = PI * 0.25 * r.y / r.x;
	} else {
		radius = r.y;
		theta = PI * 0.5 - PI * 0.25 * r.x / r.y;
	}

	return vec2(cos(theta), sin(theta)) * radius;
}

// Cosine weighted hemisphere sampling
vec3 cosine_weighted_hemisphere(vec3 normal)
{
	// Concentric disk
	vec2 r = concentric_disk();

	// Project onto hemisphere
	float z = sqrt(1.0 - r.x * r.x - r.y * r.y);
	return vec3(r.x, r.y, z) * sign(dot(normal, vec3(r.x, r.y, z)));
}
