#ifndef BOUNDING_BOX_H_
#define BOUNDING_BOX_H_

// Engine headers
#include "vec.hpp"

namespace kobra {

// Axis Aligned Bounding Box
struct BoundingBox {
	glm::vec3	min;
	glm::vec3	max;
	int		id = -1;
	
	// Get surface area of box
	float surface_area() const {
		float dx = max.x - min.x;
		float dy = max.y - min.y;
		float dz = max.z - min.z;

		float xy = dx * dy;
		float yz = dy * dz;
		float xz = dx * dz;

		return 2.0f * (xy + yz + xz);
	}
};

inline BoundingBox bbox_union(const BoundingBox &a, const BoundingBox &b)
{
	BoundingBox box;
	box.min = glm::min(a.min, b.min);
	box.max = glm::max(a.max, b.max);
	return box;
}

}

#endif
