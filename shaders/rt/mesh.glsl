#version 450

// Import bindings
#include "mesh_bindings.h"

layout (set = 0, binding = MESH_BINDING_PIXELS, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = MESH_BINDING_VIEWPORT, std430) buffer Viewport
{
	uint width;
	uint height;
} viewport;

layout (set = 0, binding = MESH_BINDING_VERTICES, std430) buffer Vertices
{
	vec4 data[];
} vertices;

layout (set = 0, binding = MESH_BINDING_TRIANGLES, std430) buffer Triangles
{
	vec4 data[];
} triangles;

// Import other headers
#include "common/color.glsl"

void main()
{
	// Offset from space origin
	uint y0 = gl_WorkGroupID.y;
	uint x0 = gl_WorkGroupID.x;

	// Return if out of bounds
	if (y0 >= viewport.height || x0 >= viewport.width)
		return;
	
	uint index = y0 * viewport.width + x0;

	// Light transport
	vec3 color = vertices.data[0].xyz + triangles.data[0].xyz;
	frame.pixels[index] = cast_color(color);
}
