#pragma once

// All possible types of objects
// TODO: refactor to primitive types
const int OBJECT_TYPE_NONE		= 0x00000000;
const int OBJECT_TYPE_TRIANGLE		= 0x00000001;
const int OBJECT_TYPE_SPHERE		= 0x00000002;

// All possible types of lights
const int LIGHT_TYPE_NONE		= 0x00000000;
const int LIGHT_TYPE_POINT		= 0x00000001;
const int LIGHT_TYPE_DIRECTIONAL	= 0x00000002;
const int LIGHT_TYPE_AREA		= 0x00000003;

// Constants
const int VERTEX_STRIDE			= 5;

enum Shading : int {
	eNone		= 0,
	eReflection	= 1 << 0,
	eTransmission	= 1 << 1,
	eDiffuse	= 1 << 2,
	eGlossy		= 1 << 3,
	eSpecular	= 1 << 4,
	eEmissive	= 1 << 5,
};

// PBR material types
// #if defined(__cplusplus) && !defined(KOBRA_OPTIX_SHADER)
//
// // Standard headers
// #include <optional>
// #include <sstream>
// #include <string>
//
// // Is a type of
// inline bool is_type(Shading type, Shading test)
// {
// 	return (type & test) == test;
// }
//
// // Convert to string
// inline std::string shading_str(Shading s)
// {
// 	std::string str;
// 	if (s & eReflection)
// 		str += "Reflection ";
// 	if (s & eTransmission)
// 		str += "Transmission ";
// 	if (s & eDiffuse)
// 		str += "Diffuse ";
// 	if (s & eGlossy)
// 		str += "Glossy ";
// 	if (s & eSpecular)
// 		str += "Specular ";
// 	if (s & eEmissive)
// 		str += "Emissive ";
//
// 	return str;
// }
//
// // Convert string to shading
// inline std::optional <Shading> shading_from_str(const std::string &str)
// {
// 	// String could contain multiple shadings
// 	std::istringstream iss(str);
//
// 	// Parse string
// 	std::string token;
//
// 	int shading = eNone;
// 	while (iss >> token) {
// 		if (token == "Reflection")
// 			shading |= eReflection;
// 		else if (token == "Transmission")
// 			shading |= eTransmission;
// 		else if (token == "Diffuse")
// 			shading |= eDiffuse;
// 		else if (token == "Glossy")
// 			shading |= eGlossy;
// 		else if (token == "Specular")
// 			shading |= eSpecular;
// 		else if (token == "Emissive")
// 			shading |= eEmissive;
// 		else
// 			return std::nullopt;
// 	}
//
// 	return static_cast <Shading> (shading);
// }
//
// #endif
