#ifndef RENDERABLE_H_
#define RENDERABLE_H_

// Engine headers
#include "material.hpp"

namespace kobra {

// Properties that every object to be
// 	rendered must have
struct Renderable {
	Material material;

	// Default constructor
	Renderable() = default;

	// Copy constructor
	Renderable(const Renderable &renderable)
			: material(renderable.material) {}

	// Constructors
	Renderable(const Material &material)
			: material(material) {}
};

}

#endif
