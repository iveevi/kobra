#ifndef KOBRA_RT_SPHERE_H_
#define KOBRA_RT_SPHERE_H_

// Engine headers
#include "../sphere.hpp"
#include "../texture_manager.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

// TODO: inherit from a general sphere class (which is an object)
class Sphere : virtual public kobra::Sphere, virtual public _element {
public:
	static constexpr char object_type[] = "RT Sphere";

	// Default constructor
	Sphere() = default;

	// Constructor
	Sphere(float radius)
			: Object(object_type, Transform()),
			kobra::Sphere(radius) {}

	Sphere(const kobra::Sphere &sphere)
			: Object(sphere.name(), object_type, sphere.transform()),
			Renderable(sphere.material),
			kobra::Sphere(sphere) {}

	// Latching to layer
	void latch(const LatchingPacket &lp, size_t id) override {
		// Offset for triangle indices
		uint offset = lp.vertices.size()/VERTEX_STRIDE;

		// Add vertex data
		// TODO: copy from a vector
		lp.vertices.push_back(glm::vec4 {_transform.position, _radius});
		lp.vertices.push_back(glm::vec4 {0.0});
		lp.vertices.push_back(glm::vec4 {0.0});
		lp.vertices.push_back(glm::vec4 {0.0});
		lp.vertices.push_back(glm::vec4 {0.0});

		// Add as an object
		uint obj_id = id - 1;

		float ia = *(reinterpret_cast <float *> (&offset));
		lp.triangles.push_back(glm::vec4 {
			ia, ia, ia,
			*(reinterpret_cast <float *> (&obj_id))
		});
		
		// Write the material
		material.serialize(lp.materials);
		
		if (material.has_albedo()) {
			auto albedo_descriptor = TextureManager::make_descriptor(
				lp.phdev, lp.device,
				material.albedo_source
			);

			lp.albedo_samplers[id - 1] = albedo_descriptor;
		}

		if (material.has_normal()) {
			auto normal_descriptor = TextureManager::make_descriptor(
				lp.phdev, lp.device,
				material.normal_source
			);

			lp.normal_samplers[id - 1] = normal_descriptor;
		}

		// Write the transform
		// TODO: do we still need this?
		lp.transforms.push_back(transform().matrix());

		// TODO: currently, sphere cannot be emmisive
	}
};

}

}

#endif
