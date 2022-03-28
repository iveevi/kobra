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
	// TODO: type of object?
	Transform _transform;
public:
	// Default constructor
	Object() = default;

	// Constructor
	Object(const Transform &transform)
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

	// Write to file
	virtual void save(std::ofstream &) const = 0;

	void save_object(std::ofstream &file) const {
		// First save transform
		file << "[TRANSFORM]" << std::endl;

		glm::vec3 position = _transform.position;
		glm::vec3 rotation = _transform.rotation;
		glm::vec3 scale = _transform.scale;

		file << "position=" << position.x << "," << position.y << "," << position.z << std::endl;
		file << "rotation=" << rotation.x << "," << rotation.y << "," << rotation.z << std::endl;
		file << "scale=" << scale.x << "," << scale.y << "," << scale.z << std::endl;

		// Save rest of object
		save(file);
	}
};

// Smart pointer
using ObjectPtr = std::shared_ptr <Object>;

}

#endif
