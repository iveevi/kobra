// Engine headers
#include "../include/renderable.hpp"
#include "../include/texture_manager.hpp"
#include "../shaders/raster/bindings.h"

namespace kobra {

// Renderable
Renderable::Renderable(const Device &dev, Mesh *mesh_)
		: mesh(mesh_)
{
	for (size_t i = 0; i < mesh->submeshes.size(); i++) {
		// Allocate memory for the vertex, index, and uniform buffers
		vk::DeviceSize vbuf_size = (*mesh)[i].vertices.size() * sizeof(Vertex);
		vk::DeviceSize ibuf_size = (*mesh)[i].indices.size() * sizeof(uint32_t);

		vertex_buffer.emplace_back(*dev.phdev, *dev.device,
			vbuf_size,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		index_buffer.emplace_back(*dev.phdev, *dev.device,
			ibuf_size,
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		ubo.emplace_back(*dev.phdev, *dev.device,
			sizeof(UBO),
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		// Upload data to buffers
		vertex_buffer[i].upload((*mesh)[i].vertices);
		index_buffer[i].upload((*mesh)[i].indices);

		// UBO
		kobra::Material mat = (*mesh)[i].material;

		UBO ubo_data {
			.diffuse = mat.diffuse,
			.specular = mat.specular,
			.emission = mat.emission,
			.ambient = mat.ambient,

			.shininess = mat.shininess,
			.roughness = mat.roughness,

			.type = static_cast <int> (mat.type),
			.has_albedo = (float) mat.has_albedo(),
			.has_normal = (float) mat.has_normal()
		};

		ubo[i].upload(&ubo_data, sizeof(UBO));

		// Other data
		index_count.push_back((*mesh)[i].indices.size());
		materials.push_back((*mesh)[i].material);
	}
}

void Renderable::draw(const vk::raii::CommandBuffer &cmd,
		const vk::raii::PipelineLayout &ppl,
		PushConstants &pc) const
{
	// TODO: parameter for which submeshes to draw
	pc.highlight = highlight ? 1.0f : 0.0f;
	for (size_t i = 0; i < materials.size(); i++) {
		// Bind and render
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*ppl, 0, *_dsets[i], {}
		);

		cmd.pushConstants <PushConstants> (*ppl,
			vk::ShaderStageFlagBits::eVertex,
			0, pc
		);

		cmd.bindVertexBuffers(0, *vertex_buffer[i].buffer, {0});
		cmd.bindIndexBuffer(*index_buffer[i].buffer, 0, vk::IndexType::eUint32);

		cmd.drawIndexed(index_count[i], 1, 0, 0, 0);
	}
}

void Renderable::bind_material
		(const Device &dev,
		const BufferData &lights_buffer,
		const std::function <vk::raii::DescriptorSet ()> &server) const
{
	_dsets.clear();
	for (int i = 0; i < materials.size(); i++) {
		_dsets.emplace_back(server());

		auto &dset = _dsets.back();
		std::string albedo = "blank";
		if (materials[i].has_albedo())
			albedo = materials[i].albedo_texture;

		std::string normal = "blank";
		if (materials[i].has_normal())
			normal = materials[i].normal_texture;

		TextureManager::bind(
			*dev.phdev, *dev.device,
			dset, albedo,
			// TODO: enum like RasterBindings::eAlbedo
			RASTER_BINDING_ALBEDO_MAP
		);

		TextureManager::bind(
			*dev.phdev, *dev.device,
			dset, normal,
			RASTER_BINDING_NORMAL_MAP
		);

		// Bind material UBO
		bind_ds(*dev.device, dset, ubo[i],
			vk::DescriptorType::eUniformBuffer,
			RASTER_BINDING_UBO
		);

		// Bind lights buffer
		bind_ds(*dev.device, dset, lights_buffer,
			vk::DescriptorType::eUniformBuffer,
			RASTER_BINDING_POINT_LIGHTS
		);
	}
}

}
