#ifndef BVH_H_
#define BVH_H_

// GLM heades
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"
#include "bbox.hpp"
#include "buffer_manager.hpp"
#include "core.hpp"
#include "world.hpp"

namespace kobra {

// C++ representation of a BVH node
struct BVHNode {
	BoundingBox	bbox;

	// Index of the object in the world
	uint		object = 0;
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
	void write(Buffer4f *buffer, int miss = -1) {
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
			leaf,
			*reinterpret_cast <float *> (&object),
			*reinterpret_cast <float *> (&hit),
			*reinterpret_cast <float *> (&miss)
		};

		// Write the node
		buffer->push_back(header);
		buffer->push_back(bbox.min);
		buffer->push_back(bbox.max);

		// Write the children
		if (left)
			left->write(buffer, miss_left);
		if (right)
			right->write(buffer, miss);
	}
};

// Aggreagate structure for BVH
struct BVH {
	// Root nodes
	std::vector <BVHNode *>	nodes;

	// Actual buffers
	Buffer4f nodes;

	// Extra information
	size_t			size = 0;
	size_t			primitives = 0;

	// Default
	BVH() {}

	// Construct BVH from bounding boxes
	BVH(const Vulkan::Context &ctx, const std::vector <BoundingBox> &bboxes) {
		// Allocate buffer
		BFM_Settings nodes_settings {
			.size = 1024,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		nodes = Buffer4(ctx, nodes_settings);

		// Get all bounding boxes
		primitives = bboxes.size();

		// Process into BVH nodes
		process(bboxes);

		// TODO: there should only be one root node remaining

		// Dump all nodes to buffer
		dump_all();
		size = dump.size() / 3;

		// Traversal stack needs at most the # of nodes
		// Quantize to 100 bytes
		size_t size = nodes.size() * sizeof(int32_t);
		size = (size + 99) / 100 * 100;

		// Map buffer
		map_buffer();
	}

	// Update BVH
	void update(const rt::World &world) {
		// Free the tree nodes
		// TODO: later conserve allocation, and
		// find a better method for dynamic BVH
		if (nodes.size() && nodes[0])
			delete nodes[0];

		// Get all bounding boxes
		std::vector <BoundingBox> boxes = world.extract_bboxes();
		primitives = boxes.size();

		// Process into BVH nodes
		process(boxes);

		// Dump all nodes to buffer
		dump_all();
		size = dump.size() / 3;

		// TODO: reallocate if needed

		// Map buffer
		map_buffer();
	}

	// Process bounding boxes into BVH nodes
	void process(const std::vector <BoundingBox> &boxes) {
		// Convert each bounding box into a BVH node
		nodes.clear();
		for (const BoundingBox &box : boxes) {
			BVHNode *node = new BVHNode(box);
			node->object = nodes.size();
			nodes.push_back(node);
		}

		// Partition nodes and single out root node
		BVHNode *root = partition(nodes);
		nodes = std::vector <BVHNode *> {root};
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

		return BoundingBox(min, max);
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
		float sa_left = BoundingBox(min_left, max_left).surface_area();
		float sa_right = BoundingBox(min_right, max_right).surface_area();
		float sa_total = BoundingBox(tmin, tmax).surface_area();

		return 1 + (prims_left * sa_left + prims_right * sa_right) / sa_total;
	}

	// Dump all nodes to buffer
	void dump_all() {
		// Clear dump
		dump.clear();

		// Dump all nodes to buffer
		for (BVHNode *node : nodes)
			node->write(dump);
	}

	// Map buffer
	void map_buffer() {
		// Map buffer
		vk->map_buffer(device,
			&buffer,
			dump.data(),
			dump.size() * sizeof(aligned_vec4)
		);

		// Stack has no need to be mapped
	}
};

}

#endif
