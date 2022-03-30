#ifndef KOBRA_RT_MESH_H_
#define KOBRA_RT_MESH_H_

// Engine headers
#include "../../shaders/rt/mesh_bindings.h"
#include "../app.hpp"
#include "../buffer_manager.hpp"
#include "../mesh.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

// Mesh for ray tracing
// TODO: inherit from primitive?
class Mesh : virtual public kobra::Mesh, virtual public _element {
	Buffer4f	_bf_vertices;
	Buffer4f	_bf_triangles;

	// Descriptor set
	VkDescriptorSet	_ds_mesh = VK_NULL_HANDLE;
public:
	static constexpr char object_type[] = "RT Mesh";

	// Default constructor
	Mesh() = default;

	// constructor from mesh
	Mesh(const App::Window &wctx, const kobra::Mesh &mesh)
			: Object(object_type, mesh.transform()),
			kobra::Mesh(mesh) {
		size_t vertices = mesh.vertex_count();
		size_t triangles = mesh.triangle_count();

		std::cout << "Mesh: " << vertices << " vertices, " << triangles << " triangles" << std::endl;

		// Allocate the buffers
		BFM_Settings vertices_settings {
			.size = vertices,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BFM_Settings triangles_settings {
			.size = triangles,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		_bf_vertices = Buffer4f(wctx.context, vertices_settings);
		_bf_triangles = Buffer4f(wctx.context, triangles_settings);

		// Iterate over the mesh and copy the data
		for (size_t i = 0; i < vertices; i++) {
			// TODO: later also push texture coordinates

			// No need to push normals, they are computed
			//	in the shader
			glm::vec3 position = _vertices[i].position;
			_bf_vertices.push_back(position);
		}

		for (size_t i = 0; i < triangles; i++) {
			uint ia = _indices[3 * i];
			uint ib = _indices[3 * i + 1];
			uint ic = _indices[3 * i + 2];

			// The shader will assume that all elements
			// 	are triangles, no need for header info:
			// 	also, material and transform
			// 	will be a push constant...
			glm::vec3 tri {
				*(reinterpret_cast <float *> (&ia)),
				*(reinterpret_cast <float *> (&ib)),
				*(reinterpret_cast <float *> (&ic))
			};

			_bf_triangles.push_back(tri);
		}

		KOBRA_LOG_FILE(notify) << "Created mesh with " << vertices
			<< " vertices and " << triangles << " triangles\n";

		std::cout << "Size of triangles: " << _bf_triangles.size()
			<< ", push_size = " << _bf_triangles.push_size() << "\n";
		
		// Flush the buffers
		_bf_vertices.sync_upload();
		_bf_triangles.sync_upload();

		KOBRA_LOG_FILE(warn) << "Uploaded mesh to GPU\n";
	}

	// Latch onto layer
	void latch_layer(const VkDescriptorSet &dset) override {
		_ds_mesh = dset;

		// Bind the buffers
		_bf_vertices.bind(_ds_mesh, MESH_BINDING_VERTICES);
		_bf_triangles.bind(_ds_mesh, MESH_BINDING_TRIANGLES);
	}

	// Get the descriptor set
	const VkDescriptorSet &dset() const override {
		return _ds_mesh;
	}

	// Virtual methods
	void render(const RenderPacket &rp) override {
		rp.pc->triangles = triangle_count();
	}
};

}

}

#endif
