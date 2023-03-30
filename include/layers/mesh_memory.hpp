#ifndef KOBRA_LAYRES_MESH_MEMORY_H_
#define KOBRA_LAYRES_MESH_MEMORY_H_

// Standard headers
#include <map>

// Engine headers
#include "../backend.hpp"
#include "../renderable.hpp"
#include "../ecs.hpp"

namespace kobra {

namespace layers {

// Contains memory relating to a renderable, about its mesh and submeshes
class MeshMemory {
public:
	// Information for a single submesh
	struct Cachelet {
		// TODO: move all the buffer datas here
	
		// CUDA mesh caches
		// TODO: combine into a contiguous array later...
		// TODO: import vertices from vulkan...
		Vertex *m_cuda_vertices = nullptr;
		glm::uvec3 *m_cuda_triangles = nullptr;
	};

	// Full information for a renderable and its mesh
	struct Cache {
		std::vector <Cachelet> m_cachelets;
	};
private:
	// Vulkan structures
	vk::raii::PhysicalDevice *m_phdev = nullptr;
	vk::raii::Device *m_device = nullptr;
	
	// TODO: macro to enable CUDA
	void fill_cachelet(Cachelet &, const Submesh &);

	// Set of all cache items
	std::map <int, Cache> m_cache;
public:
	// Default constructor
	MeshMemory() = default;

	// Constructor
	MeshMemory(const Context &context)
			: m_phdev(context.phdev), m_device(context.device) {}

	// Cache a renderable
	// void cache(const Renderable &);
	void cache_cuda(const ECS &, int);

	// Get a cache item
	const Cache &get(int entity) const {
		return m_cache.at(entity);
	}

	const Cachelet &get(int entity, size_t submesh) const {
		return m_cache.at(entity).m_cachelets.at(submesh);
	}
};

}

}

#endif
