#ifndef KOBRA_LAYER_H_
#define KOBRA_LAYER_H_

// Standard headers
#include <memory>
#include <vector>

// Engine headers
#include "scene.hpp"

namespace kobra {

// Generic layer class
template <class T>
class Layer {
public:
	// Aliases
	using ptr = std::shared_ptr <T>;
protected:
	// Elements to operate on
	std::vector <ptr> _elements;
public:
	// Default constructor
	Layer() = default;

	// Constructor from vector of elements
	Layer (const std::vector <ptr> &elements)
			: _elements (elements) {}

	// Add action
	virtual void add_do(const ptr &element) {}

	// Adding elements
	virtual void add(const ptr &element) {
		_elements.push_back(element);
		add_do(element);
	}

	virtual void add(T *element) {
		ptr p(element);
		_elements.push_back(p);
		add_do(p);
	}

	virtual void add(const std::vector <ptr> &elements) {
		_elements.insert(
			_elements.end(),
			elements.begin(),
			elements.end()
		);

		for (const auto &element : elements)
			add_do(element);
	}

	virtual void add(const std::vector <T *> &elements) {
		for (auto element : elements) {
			ptr p(element);
			_elements.push_back(p);
			add_do(p);
		}
	}

	// Adding scenes
	virtual void add_scene(const Scene &scene) = 0;

	// Indexing elements
	ptr operator[](size_t index) const {
		return _elements[index];
	}

	// Clear all elements
	virtual void clear() {
		_elements.clear();
	}

	// Number of elements
	size_t size() const {
		return _elements.size();
	}

	// TODO: variadic overload (implement collect method in common)
};

}

#endif
