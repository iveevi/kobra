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
class Raytracer;

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

// Raytracer component
// 	Wrapper around methods for raytracing
// 	a mesh entity
class Raytracer : public Renderer {
	struct HostBuffers {
		std::vector <aligned_vec4>	bvh;

		std::vector <aligned_vec4>	vertices;
		std::vector <aligned_vec4>	triangles;
		std::vector <aligned_vec4>	materials;

		std::vector <aligned_vec4>	lights;
		std::vector <uint>		light_indices;

		std::vector <aligned_mat4>	transforms;

		int id;
	};

	Mesh	*mesh = nullptr;

	// Serialize submesh data to host buffers
	void serialize_submesh(const Submesh &submesh, const Transform &transform, HostBuffers &hb) const {
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

		/* if (material.has_albedo()) {
			auto albedo_descriptor = TextureManager::make_descriptor(
				hb.phdev, lp.device,
				material.albedo_source
			);

			hb.albedo_samplers[id - 1] = albedo_descriptor;
		}

		if (material.has_normal()) {
			auto normal_descriptor = TextureManager::make_descriptor(
				hb.phdev, lp.device,
				material.normal_source
			);

			hb.normal_samplers[id - 1] = normal_descriptor;
		} */

		// Write the transform
		hb.transforms.push_back(transform.matrix());

		// NOTE: emission does not count as a light source (it will
		// still be taken care of in path tracing)
	}
public:
	// No default constructor
	Raytracer() = delete;

	// Constructor sets mesh reference
	Raytracer(Mesh *mesh_) : mesh(mesh_) {}

	// Serialize
	void serialize(const Transform &transform, HostBuffers &hb) const {
		for (size_t i = 0; i < mesh->submeshes.size(); i++)
			serialize_submesh(mesh->submeshes[i], transform, hb);

		// Increment ID per whole mesh
		hb.id++;
	}

	// TODO: build bvh

	friend class layers::Raytracer;
};

using RaytracerPtr = std::shared_ptr <Raytracer>;

}

#endif
