#ifndef BVH_H_
#define BVH_H_

// GLM heades
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"
#include "bbox.hpp"
// #include "buffer_manager.hpp"
#include "core.hpp"
// #include "world.hpp"

namespace kobra {

// C++ representation of a BVH node
// TODO: remove this implementation
struct BVHNode {
	BoundingBox	bbox;

	// Index of the object in the world
	int		object = -1;
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
	//	miss is the index of tree node
	//	that should be visited next (essential for traversal)
	void write(std::vector <aligned_vec4> &buffer, int miss = -1) {
		// Is the node a leaf?
		float leaf = (left == nullptr && right == nullptr) ? 0x1 : 0x0;

		// Current size and after
		int size = buffer.size();
		int after = size + 3;

		// Index of left and right children
		int32_t left_index = left ? after : -1;
		int32_t left_size = left ? 3 * left->size() : 0;
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
		aligned_vec4 header = glm::vec4 {
			*reinterpret_cast <float *> (&bbox.id),
			*reinterpret_cast <float *> (&object),
			*reinterpret_cast <float *> (&hit),
			*reinterpret_cast <float *> (&miss)
		};

		// Write the node
		buffer.push_back(header);
		buffer.push_back(bbox.min);
		buffer.push_back(bbox.max);

		// Write the children
		if (left)
			left->write(buffer, miss_left);
		if (right)
			right->write(buffer, miss);
	}
};

// Aggreagate structure for BVH
class BVH {
	// Root nodes
	std::vector <BVHNode *>	_nodes;
public:
	// Buffer of serliazed BVH nodes
	BufferData		buffer = nullptr;

	// Default constructor
	BVH() = default;

	// Construct BVH from bounding boxes
	BVH(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const std::vector <BoundingBox> &bboxes) {
		// Allocate buffer
		vk::DeviceSize size = 2 * 4 * bboxes.size() * sizeof(float);

		buffer = BufferData(phdev, device, size,
			vk::BufferUsageFlagBits::eStorageBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		// Process into BVH nodes
		if (bboxes.empty())
			_nodes.push_back(new BVHNode(BoundingBox()));
		else
			process(bboxes);

		// TODO: there should only be one root node remaining

		// Write nodes to buffer
		write();
	}

	// Update BVH
	void update(const std::vector <BoundingBox> &bboxes) {
		// Free the tree nodes
		if (_nodes.size() && _nodes[0])
			delete _nodes[0];

		// Process into BVH nodes
		process(bboxes);

		// Write nodes to buffer
		write();
	}

	// Process bounding boxes into BVH nodes
	void process(const std::vector <BoundingBox> &boxes) {
		// Convert each bounding box into a BVH node
		_nodes.clear();
		for (const BoundingBox &box : boxes) {
			BVHNode *node = new BVHNode(box);
			node->object = _nodes.size();
			_nodes.push_back(node);
		}

		// Partition nodes and single out root node
		BVHNode *root = partition(_nodes);
		_nodes = std::vector <BVHNode *> {root};
	}

	BVHNode *partition(std::vector <BVHNode *> &nodes) {
		// Base cases
		if (nodes.size() == 0)
			return nullptr;

		if (nodes.size() == 1)
			return nodes[0];

		if (nodes.size() == 2) {
			BVHNode *node = new BVHNode(union_of(nodes));
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
			float cost = cost_split(nodes, split, axis);

			if (cost < min_cost) {
				min_cost = cost;
				min_split = split;
			}
		}

		std::vector <BVHNode *> left;
		std::vector <BVHNode *> right;

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
			for (BVHNode *node : nodes) {
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
		BVHNode *left_node = partition(left);
		BVHNode *right_node = partition(right);

		BVHNode *node = new BVHNode(bbox);
		node->left = left_node;
		node->right = right_node;

		return node;
	}

	// Union bounding boxes
	BoundingBox union_of(const std::vector <BVHNode *> &nodes) {
		glm::vec3 min = nodes[0]->bbox.min;
		glm::vec3 max = nodes[0]->bbox.max;

		for (int i = 1; i < nodes.size(); i++) {
			min = glm::min(min, nodes[i]->bbox.min);
			max = glm::max(max, nodes[i]->bbox.max);
		}

		return BoundingBox {min, max};
	}

	float cost_split(const std::vector <BVHNode *> &nodes, float split, int axis) {
		float cost = 0.0f;

		glm::vec3 min_left = glm::vec3(std::numeric_limits <float> ::max());
		glm::vec3 max_left = glm::vec3(-std::numeric_limits <float> ::max());

		glm::vec3 min_right = glm::vec3(std::numeric_limits <float> ::max());
		glm::vec3 max_right = glm::vec3(-std::numeric_limits <float> ::max());

		glm::vec3 tmin = nodes[0]->bbox.min;
		glm::vec3 tmax = nodes[0]->bbox.max;

		int prims_left = 0;
		int prims_right = 0;

		for (BVHNode *node : nodes) {
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

	// Dump all nodes to buffer
	void write() {
		// Dump all nodes to buffer
		std::vector <aligned_vec4> host_buffer;
		for (BVHNode *node : _nodes)
			node->write(host_buffer);
		buffer.upload(host_buffer);
	}
};

}

#endif
