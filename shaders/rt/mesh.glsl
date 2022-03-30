#version 450

// Import bindings
#include "mesh_bindings.h"

layout (set = 0, binding = MESH_BINDING_PIXELS, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = MESH_BINDING_VIEWPORT, std430) buffer Vertices
{
	uint width;
	uint height;
} viewport;

// Import other headers
#include "common/color.glsl"

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	/* Return if out of bounds
	if (y0 >= world.height || x0 >= world.width)
		return; */
	
	uint index = y0 * viewport.width + x0;

	// Light transport
	vec3 color = vec3(1.0, 0.0, 1.0);

	frame.pixels[index] = cast_color(color);
}
