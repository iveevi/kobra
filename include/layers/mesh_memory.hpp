#ifndef KOBRA_LAYRES_MESH_MEMORY_H_
#define KOBRA_LAYRES_MESH_MEMORY_H_

// Standard headers
#include <map>

// Engine headers
#include "../backend.hpp"
#include "../renderable.hpp"

namespace kobra {

namespace layers {

// Contains memory relating to a renderable, about its mesh and submeshes
class MeshMemory {
	using Ref = const Renderable *;

	// Vulkan structures
	vk::raii::PhysicalDevice *m_phdev = nullptr;
	vk::raii::Device *m_device = nullptr;
	
	// TODO: macro to enable CUDA

	// Information for a single submesh
	struct Cachelet {
		// TODO: move all the buffer datas here
	
		// CUDA mesh caches
		// TODO: combine into a contiguous array later...
		glm::vec3 *m_cuda_vertices;
		
		glm::vec3 *m_cuda_normals;
		glm::vec3 *m_cuda_tangents;
		glm::vec3 *m_cuda_bitangents;
		
		glm::vec2 *m_cuda_uvs;

		glm::uvec3 *m_cuda_triangles;
	};

	// Full information for a renderable and its mesh
	struct Cache {
		std::vector <Cachelet> m_cachelets;
	};

	// Set of all cache items
	std::map <Ref, Cache> m_cache;
public:
	// Default constructor
	MeshMemory() = default;

	// Constructor
	MeshMemory(const Context &context)
			: m_phdev(context.phdev), m_device(context.device) {}

	// Cache a renderable
	// void cache(const Renderable &);
	void cache_cuda(const Renderable &);
};

}

}

#endif
