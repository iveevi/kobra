#version 450

// Glyph outlines
layout (binding = 0) buffer GlyphOutlines {
	vec2 points[];
} outlines;

// Inputs
layout(location = 0) in vec3 fcolor;
layout(location = 1) in vec2 fpos;

// Outputs
layout(location = 0) out vec4 color;

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
	return dot(p - a, normalize(norm));
}

float d2bezier(vec2 p0, vec2 p1, vec2 p2, vec2 p)
{
	float t = time(p0, p2, p);
	vec2 q0 = mix(p0, p1, t);
	vec2 q1 = mix(p1, p2, t);
	return d2line(q0, q1, p);
}

// Main function
void main()
{
	vec2 p0 = vec2(-0.808, 0.33);
	vec2 p1 = vec2(-0.807, 0.744);
	vec2 p2 = vec2(-0.083, 0.82);

	float v = d2bezier(p0, p1, p2, fpos);
	float alpha = clamp(v + 0.5, 0.0, 1.0);
	color = vec4(fcolor, alpha);
}