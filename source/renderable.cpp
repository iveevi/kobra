// Engine headers
#include "../include/renderable.hpp"
#include "../shaders/raster/bindings.h"

namespace kobra {

// Renderable
// TODO: pass subindices of the mesh to render (default = all)
Renderable::Renderable(const Context &context, Mesh *mesh_)
		: mesh(mesh_)
{
	const Device &dev = context.dev();
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
		// kobra::Material mat = (*mesh)[i].material;
		kobra::Material mat;

		int index = mesh->submeshes[i].material_index;
		if (index >= Material::all.size())
			KOBRA_LOG_FILE(Log::ERROR) << "Material index out of range\n";
		else
			mat = Material::all[index];

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
		material_indices.push_back((*mesh)[i].material_index);
	}
}

}
