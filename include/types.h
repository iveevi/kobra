#ifndef TYPES_H_
#define TYPES_H_

// All possible shading types
const float SHADING_TYPE_NONE		= 0x00000000;
const float SHADING_TYPE_FLAT		= 0x00000001;
const float SHADING_TYPE_BLINN_PHONG	= 0x00000002;

// All possible types of objects
const float OBJECT_TYPE_NONE		= 0x00000000;
const float OBJECT_TYPE_SPHERE		= 0x00000001;
const float OBJECT_TYPE_PLANE		= 0x00000002;

// All possible types of lights
const float LIGHT_TYPE_NONE		= 0x00000000;
const float LIGHT_TYPE_POINT		= 0x00000001;
const float LIGHT_TYPE_DIRECTIONAL	= 0x00000002;

#endif
