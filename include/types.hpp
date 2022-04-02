#ifndef TYPES_H_
#define TYPES_H_

// All possible shading types
const float SHADING_TYPE_NONE		= 0x00000000;
const float SHADING_TYPE_FLAT		= 0x00000001;
const float SHADING_TYPE_BLINN_PHONG	= 0x00000002;
const float SHADING_TYPE_LIGHT		= 0x00000003;	// TODO: depreciate

// Simple ray tracing with reflections and refractions
const float SHADING_TYPE_SIMPLE		= 0x00000004;

// Emmisive objects
const float SHADING_TYPE_EMMISIVE	= 0x00000005;

// All possible types of objects
// TODO: refactor to primitive types
const float OBJECT_TYPE_NONE		= 0x00000000;
const float OBJECT_TYPE_TRIANGLE	= 0x00000001;
const float OBJECT_TYPE_SPHERE		= 0x00000002;

// All possible types of lights
const float LIGHT_TYPE_NONE		= 0x00000000;
const float LIGHT_TYPE_POINT		= 0x00000001;
const float LIGHT_TYPE_DIRECTIONAL	= 0x00000002;
const float LIGHT_TYPE_AREA		= 0x00000003;

#endif
