#include "constants.h"

// Uniform buffer of lights
layout (std140, binding = 0) uniform PointLights {
	int number;

	vec3 positions[32];
} point_lights;
