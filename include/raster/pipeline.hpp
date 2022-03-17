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
Vulkan::Pipeline make_pipeline(const App::Window &wctx,
		const VkRenderPass render_pass,
		const VkShaderModule vert_shader,
		const VkShaderModule frag_shader)
{
	// Push constants
	VkPushConstantRange pcr {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(typename Mesh <T> ::MVP)
	};

	// Creation info
	Vulkan::PipelineInfo <Vertex <T> ::attributes> info {
		.swapchain = wctx.swapchain,
		.render_pass = render_pass,
		
		.vert = vert_shader,
		.frag = frag_shader,
		
		.dsls = {Vertex <T> ::descriptor_set_layout(wctx.context)},

		.vertex_binding = Vertex <T> ::vertex_binding(),
		.vertex_attributes = Vertex <T> ::vertex_attributes(),

		.push_consts = 1,
		.push_consts_range = &pcr,

		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

		.viewport {
			.width = (int) wctx.width,
			.height = - (int) wctx.height,
			.x = 0,
			.y = (int) wctx.height
		}
	};

	return wctx.context.make_pipeline(info);
}

}

}

#endif
