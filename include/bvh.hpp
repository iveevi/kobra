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
		buffer.push_back(bbox.centroid);
		buffer.push_back(bbox.dimension);

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

		// Map buffer
		map_buffer();
	}

	// Process bounding boxes into BVH nodes
	void process(const std::vector <BoundingBox> &boxes) {
		// Convert each bounding box into a BVH node
		for (const BoundingBox &box : boxes)
			nodes.push_back(new BVHNode(box));
		
		// TODO: implement partitioning
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
	}
};

}

#endif
