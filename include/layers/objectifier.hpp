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
struct Objectifier {
	// Pixels are simple 32-bit integers
	ImageData image = nullptr;
	BufferData staging_buffer = nullptr;
	DepthBuffer depth_buffer = nullptr;

	vk::raii::RenderPass render_pass = nullptr;
	vk::raii::Framebuffer framebuffer = nullptr;
	
	vk::raii::Pipeline pipeline = nullptr;
	vk::raii::PipelineLayout ppl = nullptr;
};

// TODO: an additional layer for rasterizing for the editor
// 	i.e. lights will be shown as sprites, etc

// Create an objectifier layer
Objectifier make_layer(const Context &);

// Render entities and download the image
// TODO: pack args into a struct?
void render(Objectifier &,
		const vk::raii::CommandBuffer &,
		const ECS &,
		const Camera &,
		const Transform &);

}

}

#endif
