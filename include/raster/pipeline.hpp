#ifndef PIPILINE_H_
#define PIPILINE_H_

// Engine headers
#include "mesh.hpp"
#include "../app.hpp"
#include "../texture.hpp"

// TODO: move the contents in this file to layer class

namespace kobra {

namespace raster {

// Creating and managing
// 	pipelines for each
// 	vertex type
inline Vulkan::Pipeline make_pipeline(const App::Window &wctx,
		const VkRenderPass render_pass,
		const VkShaderModule vert_shader,
		const VkShaderModule frag_shader)
{
	// Push constants
	VkPushConstantRange pcr {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(typename Mesh::MVP)
	};

	// Creation info
	Vulkan::PipelineInfo info {
		.swapchain = wctx.swapchain,
		.render_pass = render_pass,
		
		.vert = vert_shader,
		.frag = frag_shader,
		
		.dsls = {Vertex::descriptor_set_layout(wctx.context)},

		.vertex_binding = Vertex::vertex_binding(),
		.vertex_attributes = Vertex::vertex_attributes(),

		.push_consts = 1,
		.push_consts_range = &pcr,

		.depth_test = true,

		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

		.viewport {
			.width = (int) wctx.width,
			.height = (int) wctx.height,
			.x = 0,
			.y = 0
		}
	};

	return wctx.context.make_pipeline(info);
}

}

}

#endif
