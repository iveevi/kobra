#ifndef TYPES_H_
#define TYPES_H_

// All possible types of objects
const float OBJECT_TYPES[3] = {
	float(0x00000000),	// None
	float(0x00000001),	// Sphere
	float(0x00000002)	// Plane
};

// Corresponding enumerations
const int OBJT_NONE	= 0;
const int OBJT_SPHERE	= 1;
const int OBJT_PLANE	= 2;

// All possible types of lights
const float LIGHT_TYPES[3] = {
	float(0x00000000),	// None
	float(0x00000010),	// Point
	float(0x00000020)	// Directional
};

// Corresponding enumerations
const int LIGHTT_NONE	= 0;
const int LIGHTT_POINT	= 1;
const int LIGHTT_DIR	= 2;

#endif
