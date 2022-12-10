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

// Renderable component
// 	the entity must have a Mesh component
// 	TODO: reinherit from Renderer
class Renderable {
public:
	// Push constants
	struct PushConstants {
		float		time;

		alignas(16)
		glm::mat4	model;
		glm::mat4	view;
		glm::mat4	perspective;

		alignas(16)
		glm::vec3	view_position;

		// TODO: reorganize this
		float		highlight;
	};

	// Uniform buffer object
	struct UBO {
		alignas(16) glm::vec3 diffuse;
		alignas(16) glm::vec3 specular;
		alignas(16) glm::vec3 emission;
		alignas(16) glm::vec3 ambient;

		float shininess;
		float roughness;

		int type;
		float has_albedo; // TODO: encode into a single int
		float has_normal;
	};

	std::vector <BufferData>	vertex_buffer;
	std::vector <BufferData>	index_buffer;
	std::vector <BufferData>	ubo; // TODO: one single buffer, using
					     // offsets...
	
	std::vector <uint32_t>		index_count;

	// TODO: highlight should not be here
	bool				highlight = false;

	mutable std::vector <vk::raii::DescriptorSet>
					_dsets = {};
	std::vector <Material>		materials;

	// Mesh itself
	const Mesh			*mesh = nullptr;

	// Raster mode
	RasterMode mode = RasterMode::eAlbedo;

	// No default constructor
	Renderable() = delete;

	// Constructor initializes the buffers
	Renderable(const Device &, Mesh *);

	// Properties
	size_t size() const {
		return materials.size();
	}

	// Getters
	const BufferData &get_vertex_buffer(int i) const {
		return vertex_buffer[i];
	}

	const BufferData &get_index_buffer(int i) const {
		return index_buffer[i];
	}

	size_t get_index_count(int i) const {
		return index_count[i];
	}

	// Setters
	void set_highlight(bool highlight_) {
		highlight = highlight_;
	}

	// Bind resources to a descriptor set
	void draw(const vk::raii::CommandBuffer &,
		const vk::raii::PipelineLayout &ppl,
		PushConstants &) const;

	void bind_material(const Device &,
		const BufferData &,
		const std::function <vk::raii::DescriptorSet ()> &) const;
};

using RenderablePtr = std::shared_ptr <Renderable>;

}

#endif
