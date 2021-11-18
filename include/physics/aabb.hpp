#ifndef AABB_H_
#define AABB_H_

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "include/rendering.hpp"

namespace mercury {

namespace physics {

struct AxisAlignedBoundingBox {
	// NOTE: we do not bind a transform to the AABB,
	// because we do not want any rotations
	glm::vec3 center;
	glm::vec3 size;

	// Methods
	void annotate(rendering::Daemon &, Shader *);

	// Check intersecting
	bool intersects(const AxisAlignedBoundingBox &) const;
};

// Intersection method
bool intersects(const AxisAlignedBoundingBox &, const AxisAlignedBoundingBox &);

// Alias
using AABB = AxisAlignedBoundingBox;

}

}

#endif