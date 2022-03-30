#ifndef KOBRA_RT_MESH_H_
#define KOBRA_RT_MESH_H_

// Engine headers
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
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BFM_Settings triangles_settings {
			.size = triangles,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		_bf_vertices = Buffer4f(wctx.context, vertices_settings);
		_bf_vertices = Buffer4f(wctx.context, triangles_settings);

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
		// _bf_triangles.sync_upload();

		KOBRA_LOG_FILE(warn) << "Uploaded mesh to GPU\n";
	}

	// Virtual methods
	void render() override {
		KOBRA_LOG_FILE(warn) << "Mesh::render() not implemented\n";
	}
};

}

}

#endif
