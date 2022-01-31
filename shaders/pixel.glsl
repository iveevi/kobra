#version 430

layout (set = 0, binding = 0, std430) buffer Pixels
{
	int pixels[];
} frame;

layout (set = 0, binding = 1, std430) buffer World
{
	uint objects;
	uint lights;
} world;

// NOTE: pixel format is BGRA, not RGBA
void main()
{
	// Set color to red
	int color = 0xFFFFFF00;
	for (int i = 0; i < frame.pixels.length(); i++)
	{
		frame.pixels[i] = color;
	}
}
