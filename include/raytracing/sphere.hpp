#ifndef KOBRA_RT_SPHERE_H_
#define KOBRA_RT_SPHERE_H_

// Engine headers
#include "../renderable.hpp"
#include "../object.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

// TODO: inherit from a general sphere class (which is an object)
class Sphere : virtual public Object,
		virtual public Renderable,
		virtual public _element {
	glm::vec3	_center;
	float		_radius;
public:
	// Default constructor
	Sphere() = default;

	// Constructor
	Sphere(const glm::vec3& center, float radius) :
			_center(center), _radius(radius) {}

	// Latching to layer
	void latch(const LatchingPacket &lp, size_t id) override {
		// Offset for triangle indices
		uint offset = lp.vertices->push_size()/2;

		// Add vertex data
		lp.vertices->push_back(glm::vec4 {_center, _radius});
		lp.vertices->push_back(glm::vec4 {0.0});

		// Add as an object
		uint obj_id = id - 1;

		float ia = *(reinterpret_cast <float *> (&offset));
		lp.triangles->push_back(glm::vec4 {
			ia, ia, ia,
			*(reinterpret_cast <float *> (&obj_id))
		});
		
		// Write the material
		_material.write_material(lp.materials);
		
		// TODO: method for this?
		if (_material.albedo_sampler) {
			KOBRA_LOG_FILE(warn) << "Sphere has albedo texture, obj_id = " << obj_id << std::endl;
			lp.albedo_samplers[obj_id] = _material.albedo_sampler->get_image_info();
		}

		// Write the transform
		// TODO: do we still need this?
		lp.transforms->push_back(transform().matrix());

		// TODO: currently, sphere cannot be emmisive
	}

	void save(std::ofstream &file) const override {
		KOBRA_LOG_FUNC(warn) << "Sphere::save() not implemented\n";
	}
};

}

}

#endif
