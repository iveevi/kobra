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
class Mesh : virtual public kobra::Mesh, virtual public _element {
public:
	static constexpr char object_type[] = "RT Mesh";

	// Default constructor
	Mesh() = default;

	// constructor from mesh
	Mesh(const kobra::Mesh &mesh)
			: Object(object_type, mesh.transform()),
			kobra::Mesh(mesh) {
		std::cout << "Mesh: " << vertex_count() << " vertices, "
			<< triangle_count() << " triangles" << std::endl;
	}

	// Latch to layer
	void latch(const LatchingPacket &lp) override {
		// Iterate over the mesh and copy the data
		for (size_t i = 0; i < vertex_count(); i++) {
			// TODO: later also push texture coordinates

			// No need to push normals, they are computed
			//	in the shader
			glm::vec3 position = _vertices[i].position;
			lp.vertices->push_back(position);
		}

		for (size_t i = 0; i < triangle_count(); i++) {
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

			lp.triangles->push_back(tri);
		}
	}
};

}

}

#endif
