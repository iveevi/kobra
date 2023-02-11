#ifndef KOBRA_LAYERS_OBJECTIFIER_H_
#define KOBRA_LAYERS_OBJECTIFIER_H_

// Engine headers
#include "../backend.hpp"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;

namespace layers {

// This layer takes care of rendering entities
// 	into an image where we can query individual
// 	objects (i.e. for picking)
class Objectifier {
public:
	Objectifier() = default;
	Objectifier(const Context &);

	// Render entities and download the image
	// TODO: pack args into a struct?
	void render(
		const vk::raii::CommandBuffer &,
		const ECS &,
		const Camera &,
		const Transform &
	);

	// Composite a highlighting effect
	void composite_highlight(
		const vk::raii::CommandBuffer &,
		const vk::raii::Framebuffer &,
		const vk::Extent2D &,
		const ECS &,
		const Camera &,
		const Transform &,
		const std::pair <uint32_t, uint32_t> &
	);

	inline vk::Extent2D query_extent() const {
		return rendering.extent;
	}

	// Query object at a given pixel
	std::pair <uint32_t, uint32_t> query(uint32_t, uint32_t);
private:
	// Pixels are simple 32-bit integers
	struct {
		ImageData image = nullptr;
		DepthBuffer depth_buffer = nullptr;
		vk::Extent2D extent;

		vk::raii::RenderPass render_pass = nullptr;
		vk::raii::Framebuffer framebuffer = nullptr;
		
		vk::raii::Pipeline pipeline = nullptr;
		vk::raii::PipelineLayout ppl = nullptr;

		BufferData staging_buffer = nullptr;
		std::vector <uint32_t> staging_data;
	} rendering;

	struct {
		vk::raii::RenderPass render_pass = nullptr;
		vk::raii::Pipeline pipeline = nullptr;
		vk::raii::PipelineLayout ppl = nullptr;
	} compositing;
};

}

}

#endif
