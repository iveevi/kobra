#include "../../include/cuda/alloc.cuh"
#include "../../include/layers/mesh_memory.hpp"

namespace kobra {

namespace layers {

// Fill out cachelet data for a single submesh
void MeshMemory::fill_cachelet(Cachelet &cachelet, const Submesh &submesh)
{
	std::vector <glm::uvec3> triangles(submesh.triangles());

	int triangle_index = 0;
	for (int j = 0; j < submesh.indices.size(); j += 3) {
		triangles[triangle_index++] = {
			submesh.indices[j],
			submesh.indices[j + 1],
			submesh.indices[j + 2]
		};
	}

	cachelet.m_cuda_triangles = cuda::make_buffer(triangles);
	cachelet.m_cuda_vertices = cuda::make_buffer(submesh.vertices);
}

// Generate cache information for a renderable for CUDA
void MeshMemory::cache_cuda(const Entity &entity)
{
	// Check if we need to cache
	// TODO: check if the renderable has changed
	if (m_cache.find(entity.id) != m_cache.end()) {
		Cache cache = m_cache[entity.id];
		
		int count = 0;
		for (auto &cachelet : cache.m_cachelets) {
			if (cachelet.m_cuda_triangles == nullptr)
				continue;

			if (cachelet.m_cuda_vertices == nullptr)
				continue;

			count++;
		}

		if (count == cache.m_cachelets.size())
			return;
	}

        auto &renderable = entity.get <Renderable> ();
	int submeshes = renderable.size();

	std::vector <Cachelet> cachelets(submeshes);
	for (int i = 0; i < submeshes; i++) {
		// TODO: easier indexing... (make mesh private)
		// inherit renderable from shared_ptr <Mesh>?
		fill_cachelet(cachelets[i], renderable.mesh->submeshes[i]);
	}

	// Insert into cache
	Cache cache { cachelets };

	m_cache.insert({ entity.id, cache });
}

}

}
