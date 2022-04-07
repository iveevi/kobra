#ifndef OBJECT_H_
#define OBJECT_H_

// Standard headers
#include <fstream>
#include <memory>
#include <string>

// Engine headers
#include "transform.hpp"

namespace kobra {

// Scene object
class Object {
	// Generate unique names
	static int _name_id;

	static std::string _generate_name();
protected:
	std::string	_name;
	std::string	_type;
	Transform	_transform;
public:
	// Default constructor
	Object() = default;

	// Constructor
	Object(const std::string &type, const Transform &transform)
			: _type(type), _transform(transform) {
		_name = _generate_name();
	}

	Object(const std::string &name, const std::string &type, const Transform &transform)
			: _name(name), _type(type), _transform(transform) {}

	// Virtual destructor
	virtual ~Object() {}

	// Get type
	const std::string &type() const {
		return _type;
	}

	// Get name
	const std::string &name() const {
		return _name;
	}

	// Set name
	void set_name(const std::string &name) {
		_name = name;
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
		// Save name, transform, then rest of the object
		file << "[OBJECT]" << std::endl;
		file << "name=" << _name << std::endl;
		_transform.save(file);
		save(file);
	}
};

// Smart pointer
using ObjectPtr = std::shared_ptr <Object>;

}

#endif
