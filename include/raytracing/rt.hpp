#ifndef KOBRA_RT_H_
#define KOBRA_RT_H_

// Standard headers
#include <memory>

// Engine headers
#include "../backend.hpp"

namespace kobra {

namespace rt {

// Render packet
struct RenderPacket {
	VkCommandBuffer		cmd;
	VkPipelineLayout	playout;
	VkDescriptorSet		dset;
};

// Element type
struct _element {
	// Destructor
	virtual ~_element() = default;

	// Latch to layer (to get descriptor set)
	virtual void latch_layer(const VkDescriptorSet &) = 0;

	// Render
	virtual void render(const RenderPacket &) = 0;
};

// Memory safe
using Element = std::shared_ptr <_element>;

}

}

#endif
