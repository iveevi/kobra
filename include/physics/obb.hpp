#ifndef OBB_H_
#define OBB_H_

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "include/math/linalg.hpp"

namespace mercury {

// OBB class
struct OrientedBoundingBox {
        glm::vec3 center;
        glm::vec3 size;
        glm::vec3 axes[3];

        // glm::mat3 orientation;
	
        // TODO: put both these methods outside bbox, into header in physics namespace
	// Return min and max of projection on axis

        // TODO: place into source file
	Interval project(const glm::vec3 &axis) const
	{
		float min = std::numeric_limits <float> ::max();
		float max = -std::numeric_limits <float> ::max();
        
		glm::vec3 nup = glm::normalize(up);
		glm::vec3 nright = glm::normalize(right);
		glm::vec3 nforward = glm::normalize(glm::cross(nup, nright));

		glm::vec3 xdir = nright * size.x/2.0f;
		glm::vec3 ydir = nup * size.y/2.0f;
		glm::vec3 zdir = nforward * size.z/2.0f;

		std::vector <glm::vec3> pts = {
			center + xdir + ydir + zdir,
			center - xdir + ydir + zdir,
			center + xdir - ydir + zdir,
			center - xdir - ydir + zdir,
			center + xdir + ydir - zdir,
			center - xdir + ydir - zdir,
			center + xdir - ydir - zdir,
			center - xdir - ydir - zdir
		};

		for (const glm::vec3 &pt : pts) {
			float len = math::projection_length(pt, axis);
			if (len < min)
				min = len;
			if (len > max)
				max = len;
		}

		return {min, max};
	}
};

// Alias for OBB
using OBB = OrientedBoundingBox;

// TODO: move to source
/* bool intersecting(const BoundingBox &b1, const BoundingBox &b2)
{
	glm::vec3 foward1 = glm::normalize(glm::cross(b1.up, b1.right));
	glm::vec3 foward2 = glm::normalize(glm::cross(b2.up, b2.right));

	std::vector <glm::vec3> axes {  // TODO: cache axes?
		// b1 normals
		b1.right,
		b1.up,
		foward1,

		// b2 normals
		b2.right,
		b2.up,
		foward2,

		// cross products
		glm::cross(b1.right, b2.right),
		glm::cross(b1.right, b2.up),
		glm::cross(b1.right, foward2),

		glm::cross(b1.up, b2.right),
		glm::cross(b1.up, b2.up),
		glm::cross(b1.up, foward2),

		glm::cross(foward1, b2.right),
		glm::cross(foward1, b2.up),
		glm::cross(foward1, foward2)
	};

	// Iterate over all axes
	// Logger::notify() << "\tIntersection function\n";
	for (const glm::vec3 &axis : axes) {
		if (axis == glm::vec3(0, 0, 0)) {
			// Logger::notify() << "\t\tAxis is zero\n";
			continue;
		}

		Interval i1 = b1.project(axis);
		Interval i2 = b2.project(axis);

		// Logger::notify() << "\t\ti1 = " << i1.first << ", " << i1.second << "\ti2 = " << i2.first << ", " << i2.second << "\n";

		if (i1.first > i2.second || i2.first > i1.second)
			return false;
	}

	return true;
} */

}

#endif