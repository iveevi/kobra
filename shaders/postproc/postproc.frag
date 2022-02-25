#version 430

// World buffer for redner size
// TODO: import these from their modules
layout (set = 0, binding = 0, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = 1, std430) buffer World
{
	uint objects;		// TODO: this is useless
	uint primitives;
	uint lights;

	uint width;
	uint height;

	uint options;
	int discretize;	

	vec4 camera;
	vec4 cforward;
	vec4 cup;
	vec4 cright;

	vec4 tunings;

	uint indices[];
} world;

// Output color
layout (location = 0) out vec4 out_color;

// uint to vec4
vec4 u2v(uint u)
{
	return vec4(
		float((u >> 16) & 0xFF)/255.0,
		float((u >> 8) & 0xFF)/255.0,
		float(u & 0xFF)/255.0,
		1.0
	);
}

// uint at grid position
uint get_pixel(uint x, uint y)
{
	if (x >= world.width || y >= world.height)
		return 0;
	
	return frame.pixels[y * world.width + x];
}

void main()
{
	// Get the pixel buffer index
	uint x = uint(gl_FragCoord.x);
	uint y = uint(gl_FragCoord.y);

	// TODO: special case for 3x3 blur

	/* Bloom: 3x3 gaussian blur
	vec4 pixels[9] = vec4[9] (
		u2v(get_pixel(x - 1, y - 1)),
		u2v(get_pixel(x, y - 1)),
		u2v(get_pixel(x + 1, y - 1)),
		u2v(get_pixel(x - 1, y)),
		u2v(get_pixel(x, y)),
		u2v(get_pixel(x + 1, y)),
		u2v(get_pixel(x - 1, y + 1)),
		u2v(get_pixel(x, y + 1)),
		u2v(get_pixel(x + 1, y + 1))
	);

	// Average the pixels
	vec4 bloom = vec4(0.0, 0.0, 0.0, 0.0);

	vec4 m = 0.5 * vec4(1.0, 1.0, 1.0, 0.0);
	for (uint i = 0; i < 9; i++)
		out_color += clamp(pixels[i] - m, 0.0, 1.0);
	
	bloom /= 9.0;

	// Apply bloom
	out_color = pixels[4] + bloom; */

	uint blur_radius = 2;
	uint blur_width = blur_radius * 2 + 1;
	uint blur_samples = blur_width * blur_width;

	// Calculate the blur
	vec4 blur = vec4(0.0, 0.0, 0.0, 0.0);
	vec4 m = 0.7 * vec4(1.0, 1.0, 1.0, 0.0);
	for (uint i = 0; i < blur_samples; i++)
	{
		uint x = uint(gl_FragCoord.x) + uint(i % blur_width - blur_radius);
		uint y = uint(gl_FragCoord.y) + uint(i / blur_width - blur_radius);

		blur += clamp(u2v(get_pixel(x, y)) - m, 0.0, 1.0)/float(blur_samples);
	}
	
	out_color = u2v(get_pixel(x, y)) + blur;
}