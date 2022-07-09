#ifndef KOBRA_LIGHT_H_
#define KOBRA_LIGHT_H_

// Standard headers
#include <memory>

// GLM headers
#include <glm/glm.hpp>

namespace kobra {

// General light class
struct Light {
	// TODO: light type (if area light, then transform describes scale)
	// position is determined by transform
	glm::vec3 intensity;
};

using LightPtr = std::shared_ptr <Light>;

}

#endif
