#ifndef SPHERE_H_
#define SPHERE_H_

// Engine headers
#include "renderable.hpp"
#include "object.hpp"

namespace kobra {

// Basic sphere class
class Sphere : virtual public Object, virtual public Renderable {
protected:
	glm::vec3	_center;
	float		_radius;
public:
	static constexpr char object_type[] = "Sphere";

	// Default constructor
	Sphere() = default;

	// Constructor
	Sphere(const glm::vec3& center, float radius) :
			Object(object_type, Transform {center}),
			_center(center), _radius(radius) {}

	// Virtual methods
	void save(std::ofstream &file) const override {
		file << "[SPHERE]\n";
		file << "center=" << _center.x << " " << _center.y << " " << _center.z << "\n";
		file << "radius=" << _radius << "\n";
		_material.save(file);
	}
};

};

#endif

