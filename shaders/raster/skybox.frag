#version 450

// Modules
#include "bindings.h"

// Direction from camera to vertex
layout (location = 0) in vec3 dir;

// Skybox texture
layout (binding = RASTER_BINDING_SKYBOX)
uniform sampler2D skybox;

// Output color
layout (location = 0) out vec4 fragment;

// TODO: constants.glsl in shaders/
const float PI = 3.1415926535897932384626433832795;

void main()
{
	vec3 d = normalize(dir);
	float u = atan(d.x, d.z) / (2 * PI) + 0.5;
	float v = asin(d.y)/PI + 0.5;
	// fragment = texture(skybox, vec2(u, 1 - v));

	// TODO: postprocessing should be done separately...
	// fragment.xyz = pow(fragment.xyz, vec3(1.0/2.2));
}
