#ifndef KOBRA_LAYERS_GIZMO_H_
#define KOBRA_LAYERS_GIZMO_H_

// Engine headers
#include "../backend.hpp"
#include "../renderer.hpp"

namespace kobra {

// Forward declarations
class Camera;

namespace layers {

// Takes care of rendering various gizmos
struct Gizmo {
	// Type of gizmo
	enum class Type {
		eTranslate,
		eRotate,
		eScale
	};

	// Vulkan data
	vk::raii::RenderPass render_pass = nullptr;
	vk::raii::Pipeline pipeline = nullptr;
	vk::raii::PipelineLayout ppl = nullptr;
	vk::Extent2D extent;
	
	// Mesh data
	Rasterizer *translate;

	// Create a gizmo layer
	static Gizmo make(const Context &);

	// TODO: pass extent instead of storing it
	
	// Render the gizmo
	static void render(Gizmo &, Type, const Transform &,
		const vk::raii::CommandBuffer &,
		const vk::raii::Framebuffer &,
		const Camera &, const Transform &, const RenderArea &);

	// Handle dragging
	static bool handle(Gizmo &, Type, Transform &,
		const Camera &, const Transform &,
		const RenderArea &, const glm::vec2 &,
		const glm::vec2 &, bool);

	// Destroy the gizmo layer
	static void destroy(Gizmo &);
};

}

}

#endif
