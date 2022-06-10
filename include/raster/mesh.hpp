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
	BufferData		_vertex_buffer = nullptr;
	BufferData		_index_buffer = nullptr;

	// Descriptor set
	vk::raii::DescriptorSet	_dset = nullptr;
public:
	// Default constructor
	Mesh() = default;

	// Constructor
	Mesh (const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const kobra::Mesh &mesh)
			: Object(mesh.name(), object_type, mesh.transform()),
			Renderable(mesh.material().copy()),
			kobra::Mesh(mesh) {
		// Buffer sizes
		vk::DeviceSize vertex_buffer_size = _vertices.size() * sizeof(Vertex);
		vk::DeviceSize index_buffer_size = _indices.size() * sizeof(uint32_t);

		// Create buffers
		_vertex_buffer = BufferData(phdev, device,
			vertex_buffer_size,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		_index_buffer = BufferData(phdev, device,
			index_buffer_size,
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		// Upload data to buffers
		_vertex_buffer.upload(_vertices);
		_index_buffer.upload(_indices);
	}

	// Latch to layer
	void latch(const LatchingPacket &) override;

	// Add lighting info
	void light(const LightingPacket &lp) override {
		if (is_type(_material.type, eEmissive)) {
			glm::vec3 pos = center();
			lp.ubo_point_lights->positions
				[lp.ubo_point_lights->number++] = pos;
		}
	}

	// Get local descriptor set
	const vk::raii::DescriptorSet &get_local_ds() const override {
		return _dset;
	}

	// MVP structure
	struct PC_Material {
		glm::vec3	albedo;
		int		type;
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
	// TODO: call draw
	void render(RenderPacket &rp) override {
		// Get the MVP
		MVP mvp {
			_transform.matrix(),

			rp.view,
			rp.proj,

			// TODO: Material method (also keep PC_Material there)
			{
				_material.Kd,
				_material.type, // TODO: ermove this casting
				(float) rp.highlight,
				(float) _material.has_albedo(),
				(float) _material.has_normal(),
			}
		};

		// Bind the descriptor set
		rp.cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*rp.pipeline_layout, 0, {*_dset}, {}
		);

		// Bind vertices and indices
		rp.cmd.bindVertexBuffers(0, {*_vertex_buffer.buffer}, {0});
		rp.cmd.bindIndexBuffer(*_index_buffer.buffer, 0, vk::IndexType::eUint32);

		// Push constants
		rp.cmd.pushConstants <MVP> (*rp.pipeline_layout,
			vk::ShaderStageFlagBits::eVertex, 0, mvp
		);

		// Draw
		rp.cmd.drawIndexed(_indices.size(), 1, 0, 0, 0);
	}

	// Draw without descriptor set
	void draw(RenderPacket &rp) {
		// Get the MVP
		MVP mvp {
			_transform.matrix(),

			rp.view,
			rp.proj,

			// TODO: Material method (also keep PC_Material there)
			{
				_material.Kd,
				_material.type, // TODO: ermove this casting
				(float) rp.highlight,
				(float) _material.has_albedo(),
				(float) _material.has_normal(),
			}
		};

		// Bind vertices and indices
		rp.cmd.bindVertexBuffers(0, {*_vertex_buffer.buffer}, {0});
		rp.cmd.bindIndexBuffer(*_index_buffer.buffer, 0, vk::IndexType::eUint32);

		// Push constants
		rp.cmd.pushConstants <MVP> (*rp.pipeline_layout,
			vk::ShaderStageFlagBits::eVertex, 0, mvp
		);

		// Draw
		rp.cmd.drawIndexed(_indices.size(), 1, 0, 0, 0);
	}
};

}

}

#endif
