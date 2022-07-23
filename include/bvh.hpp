#ifndef KOBRA_RT_BVH_H_
#define KOBRA_RT_BVH_H_

// Standard headers
#include <memory>
#include <string>
#include <vector>

// Engine headers
#include "bbox.hpp"
#include "core.hpp"
#include "logger.hpp"

namespace kobra {

// Forward declarations
struct BVHNode;

// Shared ptr alias
using BVHPtr = std::shared_ptr <BVHNode>;

// BVH Node
// TODO: into source file!
struct BVHNode {
	// Bounding box
	BoundingBox	bbox;

	// Pointers
	int		object = -1;
	BVHPtr		left = nullptr;
	BVHPtr		right = nullptr;

	// Properties
	bool is_leaf() const;
	size_t bytes() const;
	size_t node_count() const;
	size_t primitive_count() const;
};

// Construction
BVHPtr partition(const std::vector <BVHPtr> &);
BVHPtr partition(const std::vector <BoundingBox> &);

// Serialization
void serialize(std::vector <aligned_vec4> &, const BVHPtr &, int = -1);

}

#endif
