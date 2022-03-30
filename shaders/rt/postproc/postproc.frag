#version 450

#include "../mesh_bindings.h"

// TODO: later, pass pixels as a texture sampler
layout (set = 0, binding = MESH_BINDING_PIXELS, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = MESH_BINDING_VIEWPORT, std430) buffer Vertices
{
	uint width;
	uint height;
} viewport;

// Output color
layout (location = 0) out vec4 fragment;

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
	if (x >= viewport.width || y >= viewport.height)
		return 0;
	
	return frame.pixels[y * viewport.width + x];
}

void main()
{
	// Get the pixel buffer index
	uint x = uint(gl_FragCoord.x);
	uint y = uint(gl_FragCoord.y);
	
	fragment = u2v(get_pixel(x, y));
}
