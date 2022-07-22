#ifndef KOBRA_RT_MESH_H_
#define KOBRA_RT_MESH_H_

// Engine headers
#include "../../shaders/rt/bindings.h"
#include "../app.hpp"
#include "../kmesh.hpp"
#include "../texture_manager.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

// Mesh for ray tracing
class Mesh : virtual public kobra::KMesh, virtual public _element {
public:
	static constexpr char object_type[] = "RT Mesh";

	// Default constructor
	Mesh() = default;

	// constructor from mesh
	Mesh(const kobra::KMesh &mesh)
			: Object(mesh.name(), object_type, mesh.transform()),
			Renderable(mesh.material),
			kobra::KMesh(mesh) {}

	// Latch to layer
	void latch(const LatchingPacket &lp, size_t id) override {
		// Offset for triangle indices
		uint offset = lp.vertices.size()/VERTEX_STRIDE;

		// Vertices
		// TODO: figure out how to use transform matrices in the shader
		// to apply in both bounding box and vertices
		for (size_t i = 0; i < vertex_count(); i++) {
			// TODO: later also push texture coordinates

			// No need to push normals, they are computed
			//	in the shader
			glm::vec3 position = _vertices[i].position;
			glm::vec3 normal = _vertices[i].normal;
			glm::vec3 tangent = _vertices[i].tangent;
			glm::vec3 bitangent = _vertices[i].bitangent;
			glm::vec2 uv = _vertices[i].tex_coords;

			position = _transform.apply(position);
			normal = _transform.apply_vector(normal);
			tangent = _transform.apply_vector(tangent);
			bitangent = _transform.apply_vector(bitangent);

			// TODO: premake a vector and then insert
			lp.vertices.push_back(position);
			lp.vertices.push_back(glm::vec4 {uv, 0.0f, 0.0f});
			lp.vertices.push_back(normal);
			lp.vertices.push_back(tangent);
			lp.vertices.push_back(bitangent);
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

			lp.triangles.push_back(tri);
		}

		// Write the material
		material.serialize(lp.materials);

		if (material.has_albedo()) {
			auto albedo_descriptor = TextureManager::make_descriptor(
				lp.phdev, lp.device,
				material.albedo_texture
			);

			lp.albedo_samplers[id - 1] = albedo_descriptor;
		}

		if (material.has_normal()) {
			auto normal_descriptor = TextureManager::make_descriptor(
				lp.phdev, lp.device,
				material.normal_texture
			);

			lp.normal_samplers[id - 1] = normal_descriptor;
		}

		// Write the transform
		lp.transforms.push_back(transform().matrix());

		// If the material is emmisive, write as a light
		if (is_type(material.type, eEmissive)) {
			for (size_t i = 0; i < triangle_count(); i++) {
				// Write light index
				uint index = lp.lights.size();
				lp.light_indices.push_back(index);

				uint ia = _indices[3 * i] + offset;
				uint ib = _indices[3 * i + 1] + offset;
				uint ic = _indices[3 * i + 2] + offset;

				float type = LIGHT_TYPE_AREA;

				// Header
				glm::vec4 header {
					type,
					0, 0, 0	// TODO: add intensity and other parameters
				};

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

				lp.lights.push_back(header);
				lp.lights.push_back(tri);
			}
		}
	}
};

}

}

#endif
