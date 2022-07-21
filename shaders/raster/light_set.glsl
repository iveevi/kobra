#include "constants.h"

// Light structure
// TODO: attenuation (pt and spot lights) and type
struct Light {
	vec3 position;
	vec3 intensity;
};

// Uniform buffer of lights
layout (std140, binding = RASTER_BINDING_POINT_LIGHTS) uniform Lights {
	int light_count;

	Light lights[MAX_POINT_LIGHTS];
};
