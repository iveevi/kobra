#ifndef BOUNDING_BOX_H_
#define BOUNDING_BOX_H_

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

// Axis Aligned Bounding Box
struct BoundingBox {
	glm::vec3 min;
	glm::vec3 max;

	// Constructor
	BoundingBox() {}
	BoundingBox(const glm::vec3 &a, const glm::vec3 &b)
			: min(a), max(b) {}
};

}

#endif
