#ifndef BVH_H_
#define BVH_H_

// GLM heades
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"
#include "core.hpp"

namespace mercury {

// Axis Aligned Bounding Box
struct BoundingBox {
	glm::vec3 min;
	glm::vec3 max;
};

// C++ representation of a BVH node
struct BVHNode {
	BoundingBox	bbox;
	BVHNode *	left = nullptr;
	BVHNode *	right = nullptr;

	// Constructor
	BVHNode(BoundingBox bbox) : bbox(bbox) {}

	// Destructor
	~BVHNode() {
		if (left)
			delete left;

		if (right)
			delete right;
	}

	// Get size of BVH node
	int size() {
		int size = 1;

		if (left)
			size += left->size();

		if (right)
			size += right->size();

		return size;
	}

	// Recursively write the BVH to a buffer
	void write(Buffer &buffer) {
		// TODO: implement

		// Is the node a leaf?
		float leaf = (left == nullptr && right == nullptr) ? 0x1 : 0x0;

		// Current size and after
		int size = buffer.size();
		int after = size + 3;

		// Index of left and right children
		int32_t left_index = left ? after : -1;
		int32_t left_size = left ? 3 * left->size() : 0;
		int32_t right_index = right ? after + left_size : -1;

		// Header vec4
		aligned_vec4 header = glm::vec3 {
			leaf,
			static_cast <float> (left_index),
			static_cast <float> (right_index)
		};

		// Write the node
		buffer.push_back(header);
		buffer.push_back(bbox.min);
		buffer.push_back(bbox.max);

		// Write the children
		if (left)
			left->write(buffer);
		if (right)
			right->write(buffer);
	}
};

}

#endif
