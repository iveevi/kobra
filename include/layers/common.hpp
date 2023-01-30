#ifndef KOBRA_LAYERS_COMMON_H_
#define KOBRA_LAYERS_COMMON_H_

// Engine headers
#include "../backend.hpp"

namespace kobra {

// Common structure to pass to various layers
struct RenderContext {
	const vk::raii::CommandBuffer &cmd;
	const vk::raii::Framebuffer &framebuffer;
	const vk::Extent2D &extent;
	const RenderArea &render_area = RenderArea::full();
};

}

#endif
