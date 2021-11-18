#ifndef ECS_H_
#define ECS_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

// Archetype data structure
struct Archetype {
	// Getter function:
	// 	by default returns nullptr for all types
	// 	must be specialized for each component type
	template <class Component>
	Component *get(size_t i) {
		return nullptr;
	}
};

}

#endif
