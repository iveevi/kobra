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
	Renderable(const Material &material)
			: _material(material) {}

	// Destructor
	virtual ~Renderable() {}

	// Properties
	Material &material() {
		return _material;
	}

	const Material &material() const {
		return _material;
	}
};

}

#endif
