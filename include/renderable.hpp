#ifndef KOBRA_RENDERER_H_
#define KOBRA_RENDERER_H_

// Standard headers
#include <map>

// Engine headers
#include "backend.hpp"
#include "material.hpp"
#include "mesh.hpp"

namespace kobra {

struct Renderable {
	// TODO: move to a mesh daemon that manages meshes and their vulkan resources...
	std::vector <BufferData>	vertex_buffer;
	std::vector <BufferData>	index_buffer;
	
	std::vector <uint32_t>		index_count;
	std::vector <uint32_t>		material_indices;

	// Mesh itself
        // TODO: distinguish between model and mesh:
        // renderables should contain a set of MESHES
        // that are directly indirected
	const Mesh *mesh = nullptr;

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
};

using RenderablePtr = std::shared_ptr <Renderable>;

}

#endif
