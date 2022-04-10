#ifndef KOBRA_RASTER_MESH_H_
#define KOBRA_RASTER_MESH_H_

// Engine headers
#include "../backend.hpp"
#include "../mesh.hpp"
#include "raster.hpp"

namespace kobra {

namespace raster {

// Mesh for rasterization
class Mesh : public kobra::Mesh, public _element {
public:
	static constexpr char object_type[] = "Raster Mesh";
private:
	// Vertex and index buffers
	VertexBuffer 	_vb;
	IndexBuffer	_ib;

	// Descriptor set
	VkDescriptorSet	_ds = VK_NULL_HANDLE;
public:
	// Default constructor
	Mesh() = default;

	// Constructor
	Mesh (const Vulkan::Context &ctx, const kobra::Mesh &mesh)
			: Object(mesh.name(), object_type, mesh.transform()),
			Renderable(mesh.material()),
			kobra::Mesh(mesh) {
		// Allocate vertex and index buffers
		BFM_Settings vb_settings {
			.size = this->_vertices.size(),
			.usage_type = BFM_WRITE_ONLY,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		};

		BFM_Settings ib_settings {
			.size = this->_indices.size(),
			.usage_type = BFM_WRITE_ONLY,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		};

		// TODO: alloc method for BufferManagers
		_vb = VertexBuffer(ctx, vb_settings);
		_ib = IndexBuffer(ctx, ib_settings);

		// Copy data to buffers
		_vb.push_back(this->_vertices);
		_ib.push_back(this->_indices);

		_vb.sync_size();
		_ib.sync_size();

		_vb.upload();
		_ib.upload();
	}

	// Latch to layer
	void latch(const LatchingPacket &) override;

	// Get local descriptor set
	VkDescriptorSet get_local_ds() const override {
		return _ds;
	}

	// MVP structure
	struct PC_Material {
		glm::vec3	albedo;
		float		shading_type;
		float		hightlight;
		float		has_albedo;
		float		has_normal;
	};

	struct MVP {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 projection;

		PC_Material material;
	};

	// Render
	void render(RenderPacket &rp) override {
		// Get the MVP
		MVP mvp {
			_transform.matrix(),

			rp.view,
			rp.proj,

			// TODO: Material method (also keep PC_Material there)
			{
				_material.albedo,
				_material.shading_type,
				(float) rp.highlight,
				(float) _material.has_albedo(),
				(float) _material.has_normal(),
			}
		};

		// Bind the descriptor set
		vkCmdBindDescriptorSets(rp.cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			rp.pipeline_layout,
			0, 1, &_ds,
			0, nullptr
		);

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
