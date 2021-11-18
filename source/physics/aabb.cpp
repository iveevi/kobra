#include "include/physics/aabb.hpp"

namespace mercury {

namespace physics {

// Method
bool AABB::intersects(const AABB &other) const
{
	// Edges
	glm::vec3 emax = center + size;
	glm::vec3 emin = center - size;
	
	glm::vec3 omax = other.center + other.size;
	glm::vec3 omin = other.center - other.size;

	// Check all axes
	return (omin.x <= emax.x && omax.x >= emin.x) &&
		(omin.y <= emax.y && omax.y >= emin.y) &&
		(omin.z <= emax.z && omax.z >= emin.z);
}

// Function
bool intersects(const AABB &a, const AABB &b)
{
	return a.intersects(b);
}

}

}