#ifndef BVH_H_
#define BVH_H_

// GLM heades
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"
#include "bbox.hpp"
#include "core.hpp"
#include "world.hpp"

namespace mercury {

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
		aligned_vec4 header = glm::vec4 {
			leaf,
			*reinterpret_cast <float *> (&object),
			*reinterpret_cast <float *> (&left_index),
			*reinterpret_cast <float *> (&right_index)
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

// Aggreagate structure for BVH
struct BVH {
	// Vuklan context (TODO: struct)
	Vulkan *		vk = nullptr;
	VkPhysicalDevice	phdev = VK_NULL_HANDLE;
	Vulkan::Device		device;

	// Root nodes
	std::vector <BVHNode *>	nodes;

	// Vulkan buffer
	Vulkan::Buffer		buffer;
	Vulkan::Buffer		stack; // Traversal stack
	Buffer			dump;

	// Default
	BVH() {}

	// Construct BVH from world
	BVH(Vulkan *vulkan, const VkPhysicalDevice &physical, const Vulkan::Device &dev, const World &world)
			: vk(vulkan), phdev(physical), device(dev) {
		// Get all bounding boxes
		std::vector <BoundingBox> boxes = world.extract_bboxes();

		// Process into BVH nodes
		process(boxes);

		// TODO: there should only be one root node remaining
		
		// Dump all nodes to buffer
		dump_all();

		// Allocate buffer
		vk->make_buffer(phdev, device, buffer, dump.size() * sizeof(aligned_vec4),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		// Traversal stack needs at most the # of nodes
		// Quantize to 100 bytes
		size_t size = nodes.size() * sizeof(int32_t);
		size = (size + 99) / 100 * 100;

		// Allocate stack
		vk->make_buffer(phdev, device, stack, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		// Map buffer
		map_buffer();
	}

	// Update BVH
	void update(const World &world) {
		// Free the tree nodes
		// TODO: later conserve allocation, and
		// find a better method for dynamic BVH
		if (nodes.size() && nodes[0])
			delete nodes[0];

		// Get all bounding boxes
		std::vector <BoundingBox> boxes = world.extract_bboxes();

		// Process into BVH nodes
		process(boxes);
		
		// Dump all nodes to buffer
		dump_all();

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

		Logger::ok() << "[BVH] size = " << nodes.size() << ", axis: "
			<< axis << ", (min, max) = ("
			<< min_value << ", " << max_value << ")" << std::endl;

		// Binary search optimal partition (using SAH)
		float min_cost = std::numeric_limits <float> ::max();
		float min_split = 0.0f;
		int bins = 10;

		for (int i = 0; i < bins; i++) {
			float split = (max_value - min_value) / bins * i + min_value;
			float cost = cost_split(nodes, split, axis);
			std::cout << "Candidate split: " << split << ", cost: " << cost << std::endl;

			if (cost < min_cost) {
				min_cost = cost;
				min_split = split;
			}
		}

		std::vector <BVHNode *> left;
		std::vector <BVHNode *> right;

		std::cout << "min_split = " << min_split << std::endl;
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
			
			std::cout << "\textent: " << max[axis] << " --> "
				<< min[axis] << ", median = " << value << std::endl;

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

		std::cout << "\tleft: " << prims_left << ", right: " << prims_right << std::endl;

		// Max cost when all primitives are in one side
		if (prims_left == 0 || prims_right == 0)
			return std::numeric_limits <float> ::max();

		// Compute cost
		float sa_left = BoundingBox(min_left, max_left).surface_area();
		float sa_right = BoundingBox(min_right, max_right).surface_area();
		float sa_total = BoundingBox(tmin, tmax).surface_area();

		std::cout << "\tSA_left = " << sa_left << ", SA_right = " << sa_right << std::endl;
		std::cout << "\tprims_left = " << prims_left << ", prims_right = " << prims_right << std::endl;

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
