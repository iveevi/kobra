#include "constants.h"

// Uniform buffer of lights
layout (std140, binding = RASTER_BINDING_POINT_LIGHTS) uniform PointLights {
	int number;

	vec3 positions[MAX_POINT_LIGHTS];
} point_lights;
