#ifndef KOBRA_RT_BVH_H_
#define KOBRA_RT_BVH_H_

// Standard headers
#include <memory>
#include <string>
#include <vector>

// Engine headers
#include "../bbox.hpp"
#include "../core.hpp"
#include "../logger.hpp"

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

	// Is a leaf node
	bool is_leaf() const {
		return object != -1;
	}

	// Number of nodes
	size_t node_count() const {
		return 1 + (left ? left->node_count() : 0)
			+ (right ? right->node_count() : 0);
	}

	// Number of primitives
	size_t primitive_count() const {
		return is_leaf() + (left ? left->primitive_count() : 0)
			+ (right ? right->primitive_count() : 0);
	}

	// Serialization size
	size_t bytes() const {
		return 3 * sizeof(aligned_vec4) * node_count();
	}
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
	KOBRA_ASSERT(axis >= 0 && axis < 3, "Invalid axis = " + std::to_string(axis));

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
// TODO: cost function as a parameter
inline BVHPtr partition(const std::vector <BVHPtr> &nodes)
{
	// Base cases
	if (nodes.size() == 0)
		return nullptr;

	if (nodes.size() == 1)
		return nodes[0];

	if (nodes.size() == 2) {
		BVHPtr node = std::make_shared <BVHNode> ();
		node->bbox = union_of(nodes);
		node->left = nodes[0];
		node->right = nodes[1];
		return node;
	}

	// Get axis with largest extent
	int axis = 0;
	float max_extent = 0.0f;

	float min_value = std::numeric_limits <float> ::max();
	float max_value = -std::numeric_limits <float> ::max();

	for (size_t n = 0; n < nodes.size(); n++) {
		for (int i = 0; i < 3; i++) {
			glm::vec3 min = nodes[n]->bbox.min;
			glm::vec3 max = nodes[n]->bbox.max;

			float extent = std::abs(max[i] - min[i]);
			if (extent > max_extent) {
				max_extent = extent;
				min_value = min[i];
				max_value = max[i];
				axis = i;
			}
		}
	}

	// Binary search optimal partition (using SAH)
	float min_cost = std::numeric_limits <float> ::max();
	float min_split = 0.0f;
	int bins = 10;

	for (int i = 0; i < bins; i++) {
		float split = (max_value - min_value) / bins * i + min_value;
		float cost = sah_cost(nodes, axis, split);

		if (cost < min_cost) {
			min_cost = cost;
			min_split = split;
		}
	}

	std::vector <BVHPtr> left;
	std::vector <BVHPtr> right;

	if (min_cost == std::numeric_limits <float> ::max()) {
		// Partition evenly
		for (int i = 0; i < nodes.size(); i++) {
			if (i % 2 == 0)
				left.push_back(nodes[i]);
			else
				right.push_back(nodes[i]);
		}
	} else {
		// Centroid partition with optimal split
		for (const BVHPtr &node : nodes) {
			glm::vec3 min = node->bbox.min;
			glm::vec3 max = node->bbox.max;

			float value = (min[axis] + max[axis]) / 2.0f;

			if (value < min_split)
				left.push_back(node);
			else
				right.push_back(node);
		}
	}

	// Create left and right nodes
	BoundingBox bbox = union_of(nodes);

	BVHPtr left_node = partition(left);
	BVHPtr right_node = partition(right);

	// Create parent node
	BVHPtr node = std::make_shared <BVHNode> ();
	node->bbox = bbox;
	node->left = left_node;
	node->right = right_node;

	return node;
}

// Overload with a vector of bounding boxes (convert them to BVH nodes)
inline BVHPtr partition(const std::vector <BoundingBox> &bboxes)
{
	std::vector <BVHPtr> nodes;

	for (size_t i = 0; i < bboxes.size(); i++) {
		BVHPtr node = std::make_shared <BVHNode> ();
		node->bbox = bboxes[i];
		node->object = i;
		nodes.push_back(node);
	}

	return partition(nodes);
}

// Serialize a BVH to a vector of vec4s
inline void serialize(std::vector <aligned_vec4> &buffer, const BVHPtr &bvh, int miss = -1)
{
	if (bvh == nullptr)
		return;

	// Current size and after
	int size = buffer.size();
	int after = size + 3;

	// Left and right nodes
	BVHPtr left = bvh->left;
	BVHPtr right = bvh->right;

	// Index of left and right children
	int32_t left_index = left ? after : -1;
	int32_t left_size = left ? 3 * left->node_count() : 0;
	int32_t right_index = right ? after + left_size : -1;

	// Hit index
	int32_t hit = left_index;
	if (!left && !right)
		hit = miss;

	// Miss index for left branch
	int32_t miss_left = right_index;
	if (!right)
		miss_left = miss;

	// Header vec4
	float leaf = bvh->is_leaf() ? 0x1 : 0x0;
	aligned_vec4 header = glm::vec4 {
		leaf,
		*reinterpret_cast <float *> (&bvh->object),
		*reinterpret_cast <float *> (&hit),
		*reinterpret_cast <float *> (&miss)
	};

	// Write the node
	buffer.push_back(header);
	buffer.push_back(bvh->bbox.min);
	buffer.push_back(bvh->bbox.max);

	// Write the children
	if (left)
		serialize(buffer, left, miss_left);
	if (right)
		serialize(buffer, right, miss);
}

}

}

#endif
