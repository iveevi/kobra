#ifndef ECS_H_
#define ECS_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

// Built-in components

/**
 * @brief Object: has a position
 * in the scene world
 */
struct Object {
	glm::vec3 position;
};

struct CollisionBody2D {
	Object *object;
};

struct Rigidbody2D {
	CollisionBody2D *cbody;

	void add_force(const glm::vec3 &);
	void add_torque(const glm::vec3 &);
};

// Data
struct Archetype {
	std::vector <object *>		objects;
	std::vector <CollisionBody2D *>	collisionbody2d;
	std::vector <Rigidbody2D *>	rigidbody2d;

	// Getter function:
	// 	by default returns nullptr for all types
	// 	must be specialized for each component type
	template <class Component>
	Component *get(size_t i) {
		return nullptr;
	}

	// Forward declare specializations
	template <> Object *get <Object> (size_t i);
	template <> CollisionBody2D *get <CollisionBody2D> (size_t i);
	template <> Rigidbody2D *get <Rigidbody2D> (size_t i);
};

}

#endif
