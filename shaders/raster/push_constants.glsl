#include "material.glsl"

// Push constants for vertex shaders
layout (push_constant) uniform PushConstants
{
	// Time
	float time;

	// Transform matrices
	mat4 model;
	mat4 view;
	mat4 proj;

	// Camera position
	vec3 view_pos;

	// Highlight the object?
	float highlight;
};
