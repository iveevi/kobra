// Uint color to vec3 color
// TOOO: color header
vec3 cast_color(uint c)
{
	return vec3(
		float(c & 0xFF) / 255.0,
		float((c >> 8) & 0xFF) / 255.0,
		float((c >> 16) & 0xFF) / 255.0
	);
}

// Vec3 color to uint color
uint cast_color(vec3 c)
{
	return uint(c.z * 255.0)
		| (uint(c.y * 255.0) << 8)
		| (uint(c.x * 255.0) << 16);
}

// Discretize and grey scale
vec3 discretize_grey(vec3 c, float levels)
{
	// Get gray scale
	float gray = (c.x + c.y + c.z) / 3.0;
	gray = floor(gray * levels) / levels;
	return vec3(gray, gray, gray);
}
