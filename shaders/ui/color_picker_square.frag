#version 450

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;

layout(location = 0) out vec4 fragment;

// Main function
void main()
{
	float hue = in_color.x;
	float saturation = in_uv.x;
	float lightness = (1 - in_uv.y);

	// Convert to RGB
	float chroma = (1 - abs(2 * lightness - 1)) * saturation;
	float hprime = hue * 360.0f;
	float x = chroma * (1 - abs(mod(hprime/60.0f, 2) - 1));
	float m = lightness - chroma / 2;

	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;

	if (hprime < 60.0f) {
		r = chroma;
		g = x;
	} else if (hprime < 120.0f) {
		r = x;
		g = chroma;
	} else if (hprime < 180.0f) {
		g = chroma;
		b = x;
	} else if (hprime < 240.0f) {
		g = x;
		b = chroma;
	} else if (hprime < 300.0f) {
		r = x;
		b = chroma;
	} else if (hprime < 360.0f) {
		r = chroma;
		b = x;
	}

	fragment = vec4(r + m, g + m, b + m, 1.0f);
}
