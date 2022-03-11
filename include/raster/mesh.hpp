#ifndef RASTER_MESH_H_
#define RASTER_MESH_H_

// Engine headers
#include "../mesh.hpp"
#include "raster.hpp"

namespace kobra {

namespace raster {

// Mesh for rasterization
template <VertexType T>
class Mesh : kobra::Mesh <T> {
	// Vertex and index buffers
	VertexBuffer <T>	_vb;
	IndexBuffer		_ib;
public:
	// Default constructor
	Mesh() : kobra::Mesh <T> () {}
	Mesh (const kobra::Mesh <T> &mesh) : kobra::Mesh <T> (mesh) {}

	// Render
	void render(RenderPacket &rp) {
		// Bind vertex buffer
		VkBuffer	buffers[] = {_vb.vk_buffer()};
		VkDeviceSize	offsets[] = {0};

		vkCmdBindVertexBuffers(rp.cmd, 0, 1, buffers, offsets);

		// Bind index buffer
		vkCmdBindIndexBuffer(rp.cmd, _ib.vk_buffer(), 0, VK_INDEX_TYPE_UINT32);

		// Draw
		vkCmdDrawIndexed(rp.cmd, _ib.push_size(), 1, 0, 0, 0);
	}
};

}

}

#endif
