#ifndef BOUNDING_BOX_H_
#define BOUNDING_BOX_H_

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

// Axis Aligned Bounding Box
struct BoundingBox {
	glm::vec3 centroid;
	glm::vec3 dimension;

	// Constructor
	BoundingBox() {}
	BoundingBox(const glm::vec3 &c, const glm::vec3 &d)
			: centroid(c), dimension(d) {}
};

}

#endif