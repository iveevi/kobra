#pragma once

// Engine headers
#include "transform.hpp"
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

        BoundingBox transform(const Transform &transform) {
                glm::vec3 p1 = min;
                glm::vec3 p2 = max;
                glm::vec3 p3 = glm::vec3(min.x, min.y, max.z);
                glm::vec3 p4 = glm::vec3(min.x, max.y, min.z);
                glm::vec3 p5 = glm::vec3(min.x, max.y, max.z);
                glm::vec3 p6 = glm::vec3(max.x, min.y, min.z);
                glm::vec3 p7 = glm::vec3(max.x, min.y, max.z);
                glm::vec3 p8 = glm::vec3(max.x, max.y, min.z);

                p1 = transform * p1;
                p2 = transform * p2;
                p3 = transform * p3;
                p4 = transform * p4;
                p5 = transform * p5;
                p6 = transform * p6;
                p7 = transform * p7;
                p8 = transform * p8;

                BoundingBox new_box;
                new_box.min = glm::min(glm::min(glm::min(glm::min(glm::min(glm::min(glm::min(p1, p2), p3), p4), p5), p6), p7), p8);
                new_box.max = glm::max(glm::max(glm::max(glm::max(glm::max(glm::max(glm::max(p1, p2), p3), p4), p5), p6), p7), p8);

                return new_box;
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
