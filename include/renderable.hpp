#ifndef RENDERABLE_H_
#define RENDERABLE_H_

// Engine headers
#include "material.hpp"

namespace kobra {

// Properties that every object to be
// 	rendered must have
class Renderable {
protected:
	Material _material;
public:
	// Default constructor
	Renderable() = default;

	// Constructor
	Renderable(Material &&material) {
		_material = std::move(material);
	}

	// Move constructor and assignment operator
	Renderable(Renderable &&other) = default;
	Renderable &operator=(Renderable &&other) = default;

	// Destructor
	virtual ~Renderable() {}

	// Properties
	Material &material() {
		return _material;
	}

	const Material &material() const {
		return _material;
	}

	void set_material(Material &&material) {
		_material = std::forward <Material> (material);
	}
};

}

#endif
