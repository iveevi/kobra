#ifndef PIPILINE_H_
#define PIPILINE_H_

// Engine headers
#include "../app.hpp"
#include "../vertex.hpp"

namespace kobra {

namespace raster {

// Creating and managing
// 	pipelines for each
// 	vertex type
template <VertexType T>
VkPipeline make_pipeline(const App::Window &wctx,
		const VkRenderPass render_pass,
		const VkShaderModule vert_shader,
		const VkShaderModule frag_shader)
{
	// Creation info
	Vulkan::PipelineInfo <Vertex <T> ::attributes> info {
		.swapchain = wctx.swapchain,
		.render_pass = render_pass,
		
		.vert_shader = vert_shader,
		.frag_shader = frag_shader,
		
		.dsls = Vertex <T> ::descriptor_set(wctx.descriptor_pool),

		.vertex_binding = Vertex <T> ::vertex_binding(),
		.vertex_attributes = Vertex <T> ::vertex_attributes(),

		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

		.width = wctx.width,
		.height = wctx.height
	};

	return wctx.context.make_pipeline(info);
}

}

}

#endif
