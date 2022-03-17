#ifndef RASTER_MESH_H_
#define RASTER_MESH_H_

// Engine headers
#include "../mesh.hpp"
#include "raster.hpp"

namespace kobra {

namespace raster {

// Mesh for rasterization
template <VertexType T>
class Mesh : public kobra::Mesh <T>, public _element {
	// Vertex and index buffers
	VertexBuffer <T>	_vb;
	IndexBuffer		_ib;
public:
	// Default constructor
	Mesh() : kobra::Mesh <T> () {}
	Mesh (const Vulkan::Context &ctx, const kobra::Mesh <T> &mesh) : kobra::Mesh <T> (mesh) {
		// Allocate vertex and index buffers
		BFM_Settings vb_settings {
			.size = this->_vertices.size(),
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BFM_Settings ib_settings {
			.size = this->_indices.size(),
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		// TODO: alloc method for BufferManagers
		_vb = VertexBuffer <T> (ctx, vb_settings);
		_ib = IndexBuffer(ctx, ib_settings);

		// Copy data to buffers
		_vb.push_back(this->_vertices);
		_ib.push_back(this->_indices);

		_vb.sync_size();
		_ib.sync_size();

		_vb.upload();
		_ib.upload();
	}

	// MVP structure
	struct MVP {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 projection;
	};

	// Render
	void render(RenderPacket &rp) override {
		// Get the MVP
		MVP mvp {
			this->_transform.model(),
			rp.view,
			rp.proj
		};

		// Bind vertex buffer
		VkBuffer	buffers[] = {_vb.vk_buffer()};
		VkDeviceSize	offsets[] = {0};

		vkCmdBindVertexBuffers(rp.cmd, 0, 1, buffers, offsets);

		// Bind index buffer
		vkCmdBindIndexBuffer(rp.cmd, _ib.vk_buffer(), 0, VK_INDEX_TYPE_UINT32);

		// Push constants
		vkCmdPushConstants(
			rp.cmd, rp.pipeline_layout,
			VK_SHADER_STAGE_VERTEX_BIT,
			0, sizeof(MVP), &mvp
		);

		// Draw
		vkCmdDrawIndexed(rp.cmd, _ib.push_size(), 1, 0, 0, 0);
	}
};

}

}

#endif
