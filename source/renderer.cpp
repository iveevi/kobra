// Engine headers
#include "../include/renderer.hpp"
#include "../include/texture_manager.hpp"
#include "../shaders/raster/bindings.h"

namespace kobra {

// Rasterizer
Rasterizer::Rasterizer(const Device &dev, const Mesh &mesh)
		: indices(mesh.indices())
{
	// Buffer sizes
	vk::DeviceSize vertex_buffer_size = mesh.vertices() * sizeof(Vertex);
	vk::DeviceSize index_buffer_size = mesh.indices() * sizeof(uint32_t);

	// Create buffers
	vertex_buffer = BufferData(*dev.phdev, *dev.device,
		vertex_buffer_size,
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	index_buffer = BufferData(*dev.phdev, *dev.device,
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

void Rasterizer::bind_buffers(const vk::raii::CommandBuffer &cmd) const
{
	cmd.bindVertexBuffers(0, *vertex_buffer.buffer, {0});
	cmd.bindIndexBuffer(*index_buffer.buffer, 0, vk::IndexType::eUint32);
}

void Rasterizer::bind_material(const Device &dev, const vk::raii::DescriptorSet &dset) const
{
	std::string albedo = "blank";
	if (material.has_albedo())
		albedo = material.albedo_source;

	std::string normal = "blank";
	if (material.has_normal())
		normal = material.normal_source;

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
}

// Raytracer
Raytracer::Raytracer(Mesh *mesh_) : mesh(mesh_) {}

void Raytracer::serialize_submesh(const Device &dev, const Submesh &submesh, const Transform &transform, HostBuffers &hb) const
{
	// Offset for triangle indices
	uint offset = hb.vertices.size()/VERTEX_STRIDE;

	// Vertices
	for (size_t i = 0; i < submesh.vertices.size(); i++) {
		const Vertex &v = submesh.vertices[i];

		// No need to push normals, they are computed
		//	in the shader
		glm::vec3 position = v.position;
		glm::vec3 normal = v.normal;
		glm::vec3 tangent = v.tangent;
		glm::vec3 bitangent = v.bitangent;
		glm::vec2 uv = v.tex_coords;

		position = transform.apply(position);
		normal = transform.apply_vector(normal);
		tangent = transform.apply_vector(tangent);
		bitangent = transform.apply_vector(bitangent);

		std::vector <aligned_vec4> vbuf = {
			position,
			glm::vec4 {uv, 0.0f, 0.0f},
			normal, tangent, bitangent,
		};

		hb.vertices.insert(hb.vertices.end(), vbuf.begin(), vbuf.end());
	}

	// Triangles
	uint obj_id = hb.id - 1;
	for (size_t i = 0; i < submesh.triangles(); i++) {
		uint ia = submesh.indices[3 * i] + offset;
		uint ib = submesh.indices[3 * i + 1] + offset;
		uint ic = submesh.indices[3 * i + 2] + offset;

		// The shader will assume that all elements
		// 	are triangles, no need for header info:
		// 	also, material and transform
		// 	will be a push constant...
		glm::vec4 tri {
			*(reinterpret_cast <float *> (&ia)),
			*(reinterpret_cast <float *> (&ib)),
			*(reinterpret_cast <float *> (&ic)),
			*(reinterpret_cast <float *> (&obj_id))
		};

		hb.triangles.push_back(tri);
	}

	// Write the material
	material.serialize(hb.materials);

	if (material.has_albedo()) {
		auto albedo_descriptor = TextureManager::make_descriptor(
			*dev.phdev, *dev.device,
			material.albedo_source
		);

		hb.albedo_textures[obj_id] = albedo_descriptor;
	}

	if (material.has_normal()) {
		auto normal_descriptor = TextureManager::make_descriptor(
			*dev.phdev, *dev.device,
			material.normal_source
		);

		hb.normal_textures[obj_id] = normal_descriptor;
	}

	// Write the transform
	hb.transforms.push_back(transform.matrix());

	// NOTE: emission does not count as a light source (it will
	// still be taken care of in path tracing)
}

// Serialize
void Raytracer::serialize(const Device &dev, const Transform &transform, HostBuffers &hb) const
{
	for (size_t i = 0; i < mesh->submeshes.size(); i++)
		serialize_submesh(dev, mesh->submeshes[i], transform, hb);

	// Increment ID per whole mesh
	hb.id++;
}

}
