#ifndef KOBRA_LIGHTS_H_
#define KOBRA_LIGHTS_H_

// Standard headers
#include <memory>

// GLM headers
#include <glm/glm.hpp>

namespace kobra {

// General light class
struct Light {
	enum Type {
		ePoint,
		eSpot,
		eDirectional,
		eArea // TODO: rect, disk, etc.
	};

	// Light type
	Type		type = ePoint;

	// Inteisty: color and power
	glm::vec3	color {1.0f};
	float		power {1.0f};
};

using LightPtr = std::shared_ptr <Light>;

}

#endif
