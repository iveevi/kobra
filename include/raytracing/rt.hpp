#ifndef KOBRA_RT_H_
#define KOBRA_RT_H_

// Standard headers
#include <memory>

// Engine headers
#include "../backend.hpp"

namespace kobra {

namespace rt {

// Push constant structure
struct PushConstants {
	alignas(16) uint triangles;

	aligned_vec4 camera_position;
	aligned_vec4 camera_forward;
	aligned_vec4 camera_up;
	aligned_vec4 camera_right;

	aligned_vec4 camera_tunings;
};

// Render packet
struct RenderPacket {
	PushConstants *pc;
};

// Element type
struct _element {
	// Destructor
	virtual ~_element() = default;

	// Latch to layer (to get descriptor set)
	virtual void latch_layer(const VkDescriptorSet &) = 0;

	// Get descriptor set
	virtual const VkDescriptorSet &dset() const = 0;

	// Render
	virtual void render(const RenderPacket &) = 0;
};

// Memory safe
using Element = std::shared_ptr <_element>;

}

}

#endif
