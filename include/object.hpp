#ifndef OBJECT_H_
#define OBJECT_H_

// Standard headers
#include <fstream>
#include <memory>

// Engine headers
#include "transform.hpp"

namespace kobra {

// Scene object
class Object {
protected:
	// TODO: name?
	std::string	_type;
	Transform	_transform;
public:
	// Default constructor
	Object() = default;

	// Constructor
	Object(const std::string &type, const Transform &transform)
			: _type(type), _transform(transform) {}

	// Virtual destructor
	virtual ~Object() {}

	// Get type
	const std::string &type() const {
		return _type;
	}

	// Get transform
	Transform &transform() {
		return _transform;
	}

	const Transform& transform() const {
		return _transform;
	}

	// Write to file
	virtual void save(std::ofstream &) const = 0;

	void save_object(std::ofstream &file) const {
		// Save transform, then rest of the object
		_transform.save(file);
		save(file);
	}
};

// Smart pointer
using ObjectPtr = std::shared_ptr <Object>;

}

#endif
