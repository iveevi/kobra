#ifndef KOBRA_ECS_H_
#define KOBRA_ECS_H_

// Standard headers
#include <vector>
#include <memory>

// GLM headers
#include <glm/gtx/string_cast.hpp>

// Engine headers
#include "common.hpp"
#include "logger.hpp"
#include "mesh.hpp"
#include "transform.hpp"

namespace kobra {

// Forward declarations
class Entity;

// Component to string
template <typename T>
const char *component_string()
{
	return "";
}

// Specilizations
// TODO: macrofy
template <>
inline const char *component_string <Transform> ()
{
	return "Transform";
}

template <>
inline const char *component_string <MeshPtr> ()
{
	return "Mesh";
}

// Components which all entities must have
// are stored by value
//
// The others are stored as pointers
template <class T>
using Archetype = std::vector <T>;

class ECS {
	Archetype <Transform>	transforms;
	Archetype <MeshPtr>	meshes;

	// Private helpers
	void _expand_all();
public:
	// The get functions will need to be specialized
	template <class T>
	T &get(int);

	template <class T>
	const T &get(int) const;

	// Create a new entity
	Entity make_entity(const std::string &name = "Entity");

	// Display info for one component
	template <class T>
	void info() const {
		std::cout << "Archetype: " << component_string <T> () << std::endl;
		for (size_t i = 0; i < transforms.size(); i++) {
			std::cout << "\tEntity " << i << ": ";
			if (get <T> (i) != nullptr)
				std::cout << "yes";
			else
				std::cout << "no";
			std::cout << std::endl;
		}
	}
};

// Specializations of the get functions
#define ECS_GET(T)				\
	template <>				\
	T &ECS::get <T> (int i);		\
						\
	template <>				\
	const T &ECS::get <T> (int i) const;

ECS_GET(Transform)

// Specializations of info
template <>
inline void ECS::info <Transform> () const {
	std::cout << "Archetype: " << component_string <Transform> () << std::endl;
	for (size_t i = 0; i < transforms.size(); i++) {
		std::cout << "\tEntity " << i << ": .pos = "
			<< glm::to_string(transforms[i].position) << std::endl;
	}
}

// Entity class, acts like a pointer to a component
class Entity {
	std::string	name;
	int32_t		id = -1;
	ECS		*ecs = nullptr;

	// Assert valid ECS
	void _assert() const {
		KOBRA_ASSERT(
			ecs != nullptr,
			"Entity \"" + name + "\" (id="
				+ std::to_string(id)
				+ ") has no ECS"
		);

		KOBRA_ASSERT(
			id >= 0,
			"Entity \"" + name + "\" (id="
				+ std::to_string(id)
				+ ") has invalid id"
		);
	}

	Entity(std::string name_, uint32_t id_, ECS *ecs_)
		: name(name_), id(id_), ecs(ecs_) {}
public:
	// Non copy, id is unique
	Entity(const Entity &) = delete;
	Entity &operator=(const Entity &) = delete;

	// Get for entities
	template <class T>
	T &get() {
		_assert();
		return ecs->get <T> (id);
	}

	template <class T>
	const T &get() const {
		_assert();
		return ecs->get <T> (id);
	}

	// Friend the ECS class
	friend class ECS;
};

}

#endif
