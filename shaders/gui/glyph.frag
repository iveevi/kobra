#version 450

// Glyph outlines
layout (binding = 0) buffer GlyphOutlines
{
	vec2 points[];
} outlines;

// Inputs
layout (location = 0) in vec3 fcolor;
layout (location = 1) in vec2 fpos;

// Outputs
layout (location = 0) out vec4 color;

// Bezier methods
float time(vec2 a, vec2 b, vec2 p)
{
	vec2 dir = b - a;
	float t = dot(p - a, dir) / dot(dir, dir);
	return clamp(t, 0.0, 1.0);
}

float d2line(vec2 a, vec2 b, vec2 p)
{
	vec2 dir = b - a;
	vec2 norm = vec2(-dir.y, dir.x);
	return dot(a - p, normalize(norm));
}

float d2bezier(vec2 p0, vec2 p1, vec2 p2, vec2 p, float t)
{
	vec2 q0 = mix(p0, p1, t);
	vec2 q1 = mix(p1, p2, t);
	return d2line(q0, q1, p);
}

#define BIAS 0.0001

// Main function
void main()
{
	// Get number of points
	int n = int(outlines.points[0].x);

	// Min distance to bezier
	float v = -1.0/0.0;
	float min_udist = 1.0/0.0;

	// Loop through all quadratic bezier curves
	for (int i = 1; i < n - 1; i += 2) {
		// Get points
		vec2 p0 = outlines.points[i];
		vec2 p1 = outlines.points[i + 1];
		vec2 p2 = outlines.points[i + 2];
		if (p0.x < -0.5 || p1.x < -0.5 || p2.x < -0.5) continue;

		// Time of closest point on bezier
		float t = time(p0, p2, fpos);
		float udist = distance(mix(p0, p2, t), fpos);

		if (udist <= min_udist + BIAS) {
			float bez = d2bezier(p0, p1, p2, fpos, t);

			if (udist >= min_udist - BIAS) {
				vec2 prevp = outlines.points[i - 2];
				float prevd = d2line(p0, p2, prevp);
				v = mix(min(v, bez), max(v, bez), step(prevd, 0.0));
			} else {
				v = bez;
			}

			min_udist = min(min_udist, udist);
		}
	}

	// float alpha = clamp(v + 0.5, 0.0, 1.0);
	// color = vec4(fcolor, alpha);
	// color = vec4(fpos.x, fpos.y, 0.0, 1.0);

	/* Basic bezier curve
	vec2 p0 = vec2(0.0, 0.0);
	vec2 p1 = vec2(1.0, 0.0);
	vec2 p2 = vec2(1.0, 1.0);

	float t = time(p0, p1, fpos);
	float v = d2bezier(p0, p1, p2, fpos, t); */

	float alpha = clamp(100 * v + 0.5, 0.0, 1.0);

	color = vec4(fcolor, alpha);
	if (alpha < 0.5)
		color = vec4(1.0, 0.0, 1.0, 1.0);
}
