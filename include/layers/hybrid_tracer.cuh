#ifndef KOBRA_LAYERS_HYBRID_TRACER_H_
#define KOBRA_LAYERS_HYBRID_TRACER_H_

// Engine headers
#include "../backend.hpp"

namespace kobra {

namespace layers {

// Hybrid ray/path tracer:
//	Rasterizes the scene to get the G-buffer, which is then used for ray/path
//	tracing and producing effects like GI and reflections.
struct HybridTracer {
	// Geometry buffers 
	ImageData positions = nullptr;
	ImageData normals = nullptr;

	// Material buffers
	ImageData albedo = nullptr;
	ImageData specular = nullptr;
	ImageData extra = nullptr;

	// Depth buffer
	DepthBuffer depth = nullptr;

	// Vulkan structures
	RenderPass render_pass = nullptr;
	Framebuffer framebuffer = nullptr;

	Pipeline pipeline = nullptr;
	PipelineLayout ppl = nullptr;
};

// Allocate the framebuffer images
void allocate_framebuffer_images(HybridTracer &layer, const Context &context, const vk::Extent2D &extent)
{
	// Formats for each framebuffer image
	static vk::Format fmt_positions = vk::Format::eR32G32B32Sfloat;
	static vk::Format fmt_normals = vk::Format::eR32G32B32Sfloat;
	static vk::Format fmt_albedo = vk::Format::eR32G32B32Sfloat;
	static vk::Format fmt_specular = vk::Format::eR32G32B32Sfloat;
	static vk::Format fmt_extra = vk::Format::eR32G32B32Sfloat;

	// Other image propreties
	static vk::MemoryPropertyFlags mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
	static vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
	static vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	static vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
	static vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment
		| vk::ImageUsageFlagBits::eTransferSrc;

	// Create the images
	layer.positions = ImageData {
		*context.phdev, *context.device,
		fmt_positions, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.normals = ImageData {
		*context.phdev, *context.device,
		fmt_normals, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.albedo = ImageData {
		*context.phdev, *context.device,
		fmt_albedo, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.specular = ImageData {
		*context.phdev, *context.device,
		fmt_specular, extent, tiling,
		usage, layout, mem_flags, aspect
	};

	layer.extra = ImageData {
		*context.phdev, *context.device,
		fmt_extra, extent, tiling,
		usage, layout, mem_flags, aspect
	};
}

// Create the layer
HybridTracer make_layer(const Context &context)
{
	// To return
	HybridTracer layer;

	// Create the framebuffers
	allocate_framebuffer_images(layer, context, context.extent);

	layer.depth = DepthBuffer {
		*context.phdev, *context.device,
		vk::Format::eD32Sfloat, context.extent
	};

	// Create the render pass
}

}

}

#endif
