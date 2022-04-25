#ifndef KOBRA_RT_BVH_H_
#define KOBRA_RT_BVH_H_

// Standard headers
#include <memory>
#include <vector>

// Engine headers
#include "../bbox.hpp"

namespace kobra {

namespace rt {

// Forward declarations
struct BVHNode;

// Shared ptr alias
using BVHPtr = std::shared_ptr <BVHNode>;

// BVH Node
struct BVHNode {
	// Bounding box
	BoundingBox	bbox;

	// Pointers
	int		object = -1;
	BVHPtr		left = nullptr;
	BVHPtr		right = nullptr;
};

// Union bounding box of a list of nodes
inline BoundingBox union_of(const std::vector <BVHPtr> &nodes)
{
	glm::vec3 min = nodes[0]->bbox.min;
	glm::vec3 max = nodes[0]->bbox.max;

	for (int i = 1; i < nodes.size(); i++) {
		min = glm::min(min, nodes[i]->bbox.min);
		max = glm::max(max, nodes[i]->bbox.max);
	}

	return BoundingBox {min, max};
}

// SAH cost of split
inline float sah_cost(const std::vector <BVHPtr> &nodes, int axis, float split)
{
	float cost = 0.0f;

	glm::vec3 min_left = glm::vec3(std::numeric_limits <float> ::max());
	glm::vec3 max_left = glm::vec3(-std::numeric_limits <float> ::max());

	glm::vec3 min_right = glm::vec3(std::numeric_limits <float> ::max());
	glm::vec3 max_right = glm::vec3(-std::numeric_limits <float> ::max());

	glm::vec3 tmin = nodes[0]->bbox.min;
	glm::vec3 tmax = nodes[0]->bbox.max;

	int prims_left = 0;
	int prims_right = 0;

	for (const BVHPtr &node : nodes) {
		glm::vec3 min = node->bbox.min;
		glm::vec3 max = node->bbox.max;

		tmin = glm::min(tmin, min);
		tmax = glm::max(tmax, max);

		float value = (min[axis] + max[axis]) / 2.0f;

		if (value < split) {
			// Left
			prims_left++;

			min_left = glm::min(min_left, min);
			max_left = glm::max(max_left, max);
		} else {
			// Right
			prims_right++;

			min_right = glm::min(min_right, min);
			max_right = glm::max(max_right, max);
		}
	}

	// Max cost when all primitives are in one side
	if (prims_left == 0 || prims_right == 0)
		return std::numeric_limits <float> ::max();

	// Compute cost
	float sa_left = BoundingBox {min_left, max_left}.surface_area();
	float sa_right = BoundingBox {min_right, max_right}.surface_area();
	float sa_total = BoundingBox {tmin, tmax}.surface_area();

	return 1 + (prims_left * sa_left + prims_right * sa_right) / sa_total;
}

// Partition a list of nodes
inline std::pair <std::vector <BVHPtr>, std::vector <BVHPtr>>
		partition(const std::vector <BVHPtr> &nodes)
{
}

}

}

#endif
