#version 430

layout (set = 0, binding = 0, std430) buffer Pixels
{
	uint pixels[];
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
	uint color = world.lights;
	for (int i = 0; i < 800 * 600; i++) {
		frame.pixels[i] = color;
	}
}
