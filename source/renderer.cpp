// Engine headers
#include "../include/renderer.hpp"
#include "../include/texture_manager.hpp"
#include "../shaders/raster/bindings.h"

namespace kobra {

// Rasterizer
Rasterizer::Rasterizer(const Device &dev, const Mesh &mesh, Material *mat)
		: Renderer(mat)
{
	for (size_t i = 0; i < mesh.submeshes.size(); i++) {
		// Allocate memory for the vertex and index buffers
		vk::DeviceSize vbuf_size = mesh[i].vertices.size() * sizeof(Vertex);
		vk::DeviceSize ibuf_size = mesh[i].indices.size() * sizeof(uint32_t);

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

		// Upload data to buffers
		vertex_buffer[i].upload(mesh[i].vertices);
		index_buffer[i].upload(mesh[i].indices);
		index_count.push_back(mesh[i].indices.size());
		materials.push_back(mesh[i].material);
	}
}

void Rasterizer::draw(const vk::raii::CommandBuffer &cmd,
		const vk::raii::PipelineLayout &ppl,
		PushConstants &pc) const
{
	// TODO: parameter for whih submeshes to draw
	pc.albedo = material->diffuse;
	pc.type = Shading::eDiffuse;
	pc.has_albedo = material->has_albedo();
	pc.has_normal = material->has_normal();
	
	for (size_t i = 0; i < vertex_buffer.size(); i++) {
		// Set the push constants
		pc.albedo = materials[i].diffuse;

		// Render
		cmd.pushConstants <PushConstants> (*ppl,
			vk::ShaderStageFlagBits::eVertex,
			0, pc
		);

		cmd.bindVertexBuffers(0, *vertex_buffer[i].buffer, {0});
		cmd.bindIndexBuffer(*index_buffer[i].buffer, 0, vk::IndexType::eUint32);

		cmd.drawIndexed(index_count[i], 1, 0, 0, 0);
	}
}

void Rasterizer::bind_material(const Device &dev, const vk::raii::DescriptorSet &dset) const
{
	std::string albedo = "blank";
	if (material->has_albedo())
		albedo = material->albedo_texture;

	std::string normal = "blank";
	if (material->has_normal())
		normal = material->normal_texture;

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
Raytracer::Raytracer(Mesh *mesh_, Material *material_)
		: Renderer(material_), mesh(mesh_) {}

const Mesh &Raytracer::get_mesh() const
{
	return *mesh;
}

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
	_material smat;
	smat.diffuse = material->diffuse;
	smat.specular = material->specular;
	smat.emission = material->emission;
	smat.ambient = material->ambient;
	smat.shininess = material->shininess;
	smat.roughness = material->roughness;
	smat.refraction = material->refraction;
	smat.albedo = material->has_albedo();
	smat.normal = material->has_normal();
	smat.type = material->type;

	hb.materials.push_back(smat);

	if (material->has_albedo()) {
		auto albedo_descriptor = TextureManager::make_descriptor(
			*dev.phdev, *dev.device,
			material->albedo_texture
		);

		hb.albedo_textures[obj_id] = albedo_descriptor;
	}

	if (material->has_normal()) {
		auto normal_descriptor = TextureManager::make_descriptor(
			*dev.phdev, *dev.device,
			material->normal_texture
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
