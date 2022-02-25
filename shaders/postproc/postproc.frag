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

void main()
{
	// Get the pixel buffer index
	uint x = uint(gl_FragCoord.x);
	uint y = uint(gl_FragCoord.y);
	uint index = x + y * world.width;

	// Return pixel at index
	uint pixel = frame.pixels[index];

	// Extract color channels
	out_color.x = float((pixel >> 16) & 0xFF) / 255.0;
	out_color.y = float((pixel >> 8) & 0xFF) / 255.0;
	out_color.z = float(pixel & 0xFF) / 255.0;
	out_color.w = 1.0;
}