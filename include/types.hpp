#ifndef TYPES_H_
#define TYPES_H_

// TODO: refactor this header

// All possible shading types
const float SHADING_TYPE_NONE		= 0x00000000;
const float SHADING_TYPE_FLAT		= 0x00000001;
const float SHADING_TYPE_BLINN_PHONG	= 0x00000002;
const float SHADING_TYPE_LIGHT		= 0x00000003;	// TODO: depreciate

// Simple ray tracing with reflections and refractions
const float SHADING_TYPE_SIMPLE		= 0x00000004;

// Emmisive objects
const float SHADING_TYPE_EMISSIVE	= 0x00000005;

// Raytracing shading types
const float SHADING_TYPE_DIFFUSE	= 1 << 4;
const float SHADING_TYPE_SPECULAR	= 1 << 5;
const float SHADING_TYPE_REFLECTION	= 1 << 6;
const float SHADING_TYPE_REFRACTION	= 1 << 7;

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

// Constants
const int VERTEX_STRIDE			= 5;

#endif
