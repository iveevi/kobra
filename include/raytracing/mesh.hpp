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
	void latch(const LatchingPacket &lp, size_t id) override {
		// Offset for triangle indices
		uint offset = lp.vertices->push_size();

		// Vertices
		// TODO: figure out how to use transform matrices in the shader
		// to apply in both bounding box and vertices
		for (size_t i = 0; i < vertex_count(); i++) {
			// TODO: later also push texture coordinates

			// No need to push normals, they are computed
			//	in the shader
			glm::vec3 position = _vertices[i].position;
			glm::vec2 uv = _vertices[i].tex_coords;

			position = _transform.apply(position);

			lp.vertices->push_back(position);
			lp.vertices->push_back(glm::vec4 {uv, 0.0f, 0.0f});
		}

		// Triangles
		uint obj_id = id - 1;
		for (size_t i = 0; i < triangle_count(); i++) {
			uint ia = _indices[3 * i] + offset;
			uint ib = _indices[3 * i + 1] + offset;
			uint ic = _indices[3 * i + 2] + offset;

			// The shader will assume that all elements
			// 	are triangles, no need for header info:
			// 	also, material and transform
			// 	will be a push constant...
			glm::vec4 tri {
				*(reinterpret_cast <float *> (&ia)),
				*(reinterpret_cast <float *> (&ib)),
				*(reinterpret_cast <float *> (&ic)),
				*(reinterpret_cast <float *> (&obj_id))
			};

			lp.triangles->push_back(tri);
		}

		// Write the material
		_material.write_material(lp.materials);

		// Write the transform
		lp.transforms->push_back(transform().matrix());
	}
};

}

}

#endif
