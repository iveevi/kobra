#ifndef KOBRA_LAYER_H_
#define KOBRA_LAYER_H_

// Standard headers
#include <memory>
#include <vector>

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

	// Adding elements
	void add(const ptr &element) {
		_elements.push_back (element);
	}

	void add(T *element) {
		_elements.push_back(ptr(element));
	}

	void add(const std::vector <ptr> &elements) {
		_elements.insert(
			_elements.end(),
			elements.begin(),
			elements.end()
		);
	}

	void add(const std::vector <T *> &elements) {
		for (auto element : elements)
			_elements.push_back(ptr(element));
	}

	// TODO: variadic overload (implement collect method in common)
	
	// TODO: virtual render method
};

}

#endif
