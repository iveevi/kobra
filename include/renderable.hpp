#ifndef KOBRA_RENDERER_H_
#define KOBRA_RENDERER_H_

// Standard headers
#include <map>

// Engine headers
#include "backend.hpp"
#include "material.hpp"
#include "mesh.hpp"

namespace kobra {

// Renderable component:
// 	Consists of a reference to a mesh
// 	and its underlying data
class Renderable {
public:
	// Uniform buffer object
	// struct UBO {
	// 	alignas(16) glm::vec3 diffuse;
	// 	alignas(16) glm::vec3 specular;
	// 	alignas(16) glm::vec3 emission;
	// 	alignas(16) glm::vec3 ambient;
	//
	// 	float shininess;
	// 	float roughness;
	//
	// 	int type;
	// 	float has_albedo; // TODO: encode into a single int
	// 	float has_normal;
	// };

	// TODO: move to another layer...
	std::vector <BufferData>	vertex_buffer;
	std::vector <BufferData>	index_buffer;
	// std::vector <BufferData>	ubo; // TODO: one single buffer, using
					     // offsets...
	
	std::vector <uint32_t>		index_count;
	std::vector <uint32_t>		material_indices;

	// Mesh itself
        // TODO: distinguish between model and mesh:
        // renderables should contain a set of MESHES
        // that are directly indirected
	const Mesh			*mesh = nullptr;

	// No default or copy constructor
	Renderable() = delete;
	Renderable(const Renderable &) = delete;
	const Renderable &operator=(const Renderable &) = delete;

	// Constructor initializes the buffers
	Renderable(const Context &, Mesh *);

	// Properties
	size_t size() const {
		return material_indices.size();
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
};

using RenderablePtr = std::shared_ptr <Renderable>;

}

#endif
