#ifndef KOBRA_RASTER_MESH_H_
#define KOBRA_RASTER_MESH_H_

// Engine headers
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
	void latch(const LatchingPacket &lp) override {
		// Only do stuff if the mesh is emissive
		if (_material.shading_type != SHADING_TYPE_EMISSIVE)
			return;

		KOBRA_LOG_FUNC(notify) << "Latching emissive mesh\n";
		glm::vec3 pos = _transform.apply(centroid());
		lp.ubo_point_lights->positions
			[lp.ubo_point_lights->number++] = pos;
	}

	// MVP structure
	struct PC_Material {
		glm::vec3	albedo;
		float		shading_type;
		float		hightlight;
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
			}
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
