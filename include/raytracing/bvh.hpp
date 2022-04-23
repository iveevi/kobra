#ifndef KOBRA_RT_BVH_H_
#define KOBRA_RT_BVH_H_

// Standard headers
#include <memory>

// Engine headers
#include "../bbox.hpp"

namespace kobra {

namespace rt {

// BVH Node
struct BVHNode {
	// Bounding box
	BoundingBox	bbox;

	// Pointers
	int		object = -1;
	BVHNode		*left = nullptr;
	BVHNode		*right = nullptr;

	// Destructor
	~BVHNode() {
		if (left)
			delete left;

		if (right)
			delete right;
	}
};

// Shared ptr alias
using BVHPtr = std::shared_ptr <BVHNode>;

// Create BVH from a list of bounding boxes

}

}

#endif
