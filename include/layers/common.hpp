#ifndef KOBRA_LAYERS_COMMON_H_
#define KOBRA_LAYERS_COMMON_H_

// Engine headers
#include "../backend.hpp"

namespace kobra {

// Common structure to pass to various layers
struct RenderContext {
	const vk::raii::CommandBuffer &cmd;
	const vk::raii::Framebuffer &framebuffer;
	vk::Extent2D extent;
	RenderArea render_area = RenderArea::full();

	RenderContext cropped(const RenderArea &area) const {
		RenderContext ctx = *this;
		ctx.render_area = area;
		return ctx;
	}
};

}

#endif
