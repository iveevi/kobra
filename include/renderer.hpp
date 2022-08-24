#ifndef KOBRA_RENDERER_H_
#define KOBRA_RENDERER_H_

// Standard headers
#include <map>

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
	// TODO: make private
	// TODO: should hold a list of materials...
	Material *material = nullptr;

	// No default constructor
	Renderer() = delete;

	// Constructor
	Renderer(Material *material_) : material(material_) {}

	const Material &get_material() const {
		return *material;
	}
};

// Rasterizer component
// 	the entity must have a Mesh component
class Rasterizer : public Renderer {
	// Push constants
	struct PushConstants {
		glm::mat4	model;
		glm::mat4	view;
		glm::mat4	perspective;

		glm::vec3	albedo;
		int		type;
		float		highlight;
		float		has_albedo;
		float		has_normal;
	};
	
	using ResourceMap = std::map <const Rasterizer *, vk::raii::DescriptorSet>;

	std::vector <BufferData>	vertex_buffer;
	std::vector <BufferData>	index_buffer;
	std::vector <uint32_t>		index_count;
	std::vector <Material>		materials;

	mutable std::vector <vk::raii::DescriptorSet>
					_dsets = {};
public:
	// Raster mode
	RasterMode mode = RasterMode::eAlbedo;

	// No default constructor
	Rasterizer() = delete;

	// Constructor initializes the buffers
	Rasterizer(const Device &, const Mesh &, Material *);

	// Bind resources to a descriptor set
	void draw(const vk::raii::CommandBuffer &,
		const vk::raii::PipelineLayout &ppl,
		PushConstants &) const;

	void bind_material(const Device &,
		const BufferData &,
		const std::function <vk::raii::DescriptorSet ()> &) const;

	// Friends
	friend class layers::Raster;
};

using RasterizerPtr = std::shared_ptr <Rasterizer>;

// Raytracer component
// 	Wrapper around methods for raytracing
// 	a mesh entity
class Raytracer : public Renderer {
	struct alignas(16) _material {
		alignas(16) glm::vec3 diffuse;
		alignas(16) glm::vec3 specular;
		alignas(16) glm::vec3 emission;
		alignas(16) glm::vec3 ambient;
		float shininess;
		float roughness;
		float refraction;
		int albedo;
		int normal;
		int type;
	};

	struct HostBuffers {
		std::vector <aligned_vec4>	bvh;

		std::vector <aligned_vec4>	vertices;
		std::vector <aligned_vec4>	triangles;
		std::vector <_material>		materials;

		std::vector <aligned_vec4>	lights;
		std::vector <uint>		light_indices;

		std::vector <aligned_mat4>	transforms;

		std::vector <vk::DescriptorImageInfo> &albedo_textures;
		std::vector <vk::DescriptorImageInfo> &normal_textures;

		int id;
	};

	Mesh	*mesh = nullptr;

	// Serialize submesh data to host buffers
	void serialize_submesh(const Device &dev, const Submesh &, const Transform &, HostBuffers &) const;
public:
	// No default constructor
	Raytracer() = delete;

	// Constructor sets mesh reference
	Raytracer(Mesh *, Material *);

	// Get the mesh
	const Mesh &get_mesh() const;

	// Serialize
	void serialize(const Device &, const Transform &, HostBuffers &) const;

	// TODO: build bvh

	friend class layers::Raytracer;
};

using RaytracerPtr = std::shared_ptr <Raytracer>;

}

#endif
