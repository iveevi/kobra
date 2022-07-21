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
	Rasterizer(const Device &, const Mesh &);

	// Bind resources to a descriptor set
	void bind_buffers(const vk::raii::CommandBuffer &) const;
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
	void serialize_submesh(const Submesh &, const Transform &, HostBuffers &) const;
public:
	// No default constructor
	Raytracer() = delete;

	// Constructor sets mesh reference
	Raytracer(Mesh *);

	// Serialize
	void serialize(const Transform &, HostBuffers &) const;

	// TODO: build bvh

	friend class layers::Raytracer;
};

using RaytracerPtr = std::shared_ptr <Raytracer>;

}

#endif
