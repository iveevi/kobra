#ifndef OBJECT_H_
#define OBJECT_H_

#include "transform.hpp"

namespace kobra {

// Scene object
class Object {
protected:
	Transform _transform;
public:
	// Constructor
	Object(const Transform& transform = Transform())
		: _transform(transform) {}

	// Virtual destructor
	virtual ~Object() {}

	// Get transform
	Transform &transform() {
		return _transform;
	}

	const Transform& transform() const {
		return _transform;
	}
};

}

#endif
