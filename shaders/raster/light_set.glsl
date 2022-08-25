#include "constants.h"

// Light structure
// TODO: attenuation (pt and spot lights) and type
struct Light {
	vec3 position;
	vec3 intensity;
};

struct AreaLight {
	vec3 a;
	vec3 ab;
	vec3 ac;
	vec3 intensity;
};

// Uniform buffer of lights
layout (std140, binding = RASTER_BINDING_POINT_LIGHTS) uniform Lights {
	int light_count;
	int n_area_lights;

	Light lights[MAX_POINT_LIGHTS];
	AreaLight area_lights[MAX_POINT_LIGHTS];
};
