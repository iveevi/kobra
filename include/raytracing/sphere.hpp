#ifndef KOBRA_RT_SPHERE_H_
#define KOBRA_RT_SPHERE_H_

// Engine headers
#include "../sphere.hpp"
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
			Renderable(sphere.material()),
			kobra::Sphere(sphere) {}

	// Latching to layer
	void latch(const LatchingPacket &lp, size_t id) override {
		// Offset for triangle indices
		uint offset = lp.vertices->push_size()/VERTEX_STRIDE;

		// Add vertex data
		lp.vertices->push_back(glm::vec4 {_transform.position, _radius});
		lp.vertices->push_back(glm::vec4 {0.0});
		lp.vertices->push_back(glm::vec4 {0.0});
		lp.vertices->push_back(glm::vec4 {0.0});
		lp.vertices->push_back(glm::vec4 {0.0});

		// Add as an object
		uint obj_id = id - 1;

		float ia = *(reinterpret_cast <float *> (&offset));
		lp.triangles->push_back(glm::vec4 {
			ia, ia, ia,
			*(reinterpret_cast <float *> (&obj_id))
		});
		
		// Write the material
		_material.serialize(lp.materials);
		
		auto albedo_descriptor = _material.get_albedo_descriptor();
		if (albedo_descriptor)
			lp.albedo_samplers[id - 1] = albedo_descriptor.value();

		auto normal_descriptor = _material.get_normal_descriptor();
		if (normal_descriptor)
			lp.normal_samplers[id - 1] = normal_descriptor.value();

		// Write the transform
		// TODO: do we still need this?
		lp.transforms->push_back(transform().matrix());

		// TODO: currently, sphere cannot be emmisive
	}
};

}

}

#endif
