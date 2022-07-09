#ifndef KOBRA_RENDERER_H_
#define KOBRA_RENDERER_H_

// Engine headers
#include "backend.hpp"
#include "enums.hpp"
#include "material.hpp"
#include "mesh.hpp"

namespace kobra {

// Forward declarations
namespace layers {

class Raster;

}

// Handles the rendering of an entity
struct Renderer {
	Material material;
};

// Rasterizer component
// 	the entity must have a Mesh component
class Rasterizer : public Renderer {
	BufferData	vertex_buffer = nullptr;
	BufferData	index_buffer = nullptr;
	size_t		indices = 0;
public:
	// Raster mode
	RasterMode mode = RasterMode::eAlbedo;

	// No default constructor
	Rasterizer() = delete;

	// Constructor initializes the buffers
	Rasterizer(const Device &dev, const Mesh &mesh)
			: indices(mesh.indices()) {
		// Buffer sizes
		vk::DeviceSize vertex_buffer_size = mesh.vertices() * sizeof(Vertex);
		vk::DeviceSize index_buffer_size = mesh.indices() * sizeof(uint32_t);

		// Create buffers
		vertex_buffer = BufferData(dev.phdev, dev.device,
			vertex_buffer_size,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		index_buffer = BufferData(dev.phdev, dev.device,
			index_buffer_size,
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		vk::DeviceSize voffset = 0;
		vk::DeviceSize ioffset = 0;

		for (size_t i = 0; i < mesh.submeshes.size(); i++) {
			vk::DeviceSize vbuf_size = mesh[i].vertices.size() * sizeof(Vertex);
			vk::DeviceSize ibuf_size = mesh[i].indices.size() * sizeof(uint32_t);

			// Upload data to buffers
			vertex_buffer.upload(mesh[i].vertices, voffset);
			index_buffer.upload(mesh[i].indices, ioffset);

			// Increment offsets
			voffset += vbuf_size;
			ioffset += ibuf_size;
		}
	}

	// Bind resources to a descriptor set
	void bind_buffers(const vk::raii::CommandBuffer &cmd) const {
		cmd.bindVertexBuffers(0, *vertex_buffer.buffer, {0});
		cmd.bindIndexBuffer(*index_buffer.buffer, 0, vk::IndexType::eUint32);
	}

	void bind_material(const Device &, const vk::raii::DescriptorSet &) const;

	// Friends
	friend class layers::Raster;
};

using RasterizerPtr = std::shared_ptr <Rasterizer>;

}

#endif
