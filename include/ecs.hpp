#ifndef KOBRA_ECS_H_
#define KOBRA_ECS_H_

// Standard headers
#include <vector>
#include <memory>

// GLM headers
#include <glm/gtx/string_cast.hpp>

// Engine headers
#include "camera.hpp"
#include "common.hpp"
#include "logger.hpp"
#include "mesh.hpp"
#include "renderer.hpp"
#include "transform.hpp"

namespace kobra {

// Forward declarations
class Entity;

// Component to string
template <typename T>
std::string component_string()
{
	return "";
}

// Specilizations
// TODO: macrofy
template <>
inline std::string component_string <Transform> ()
{
	return "Transform";
}

template <>
inline std::string component_string <MeshPtr> ()
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
	Archetype <Transform>		transforms;
	Archetype <MeshPtr>		meshes;
	Archetype <RasterizerPtr>	rasterizers;
	Archetype <CameraPtr>		cameras;

	// Private helpers
	void _expand_all();

	// For construction of components
	template <class T>
	struct _constructor {
		template <class ... Args>
		static T make(Args ... args) {
			return T(args ...);
		}
	};

	// For referencing archetypes (as a whole)
	// TODO: combine with _constructor
	template <class T>
	struct _ref {
		static T &ref(ECS *, int) {
			throw std::runtime_error(
				"_ref::ref() not implemented for type: "
				+ component_string <T> ()
			);
		}

		static T &get(ECS *, int) {
			throw std::runtime_error(
				"_ref::get() not implemented for type: "
				+ component_string <T> ()
			);
		}

		static const T &get(const ECS *, int) {
			throw std::runtime_error(
				"const _ref::get() not implemented for type: "
				+ component_string <T> ()
			);
		}

		static bool exists(const ECS *, int) {
			throw std::runtime_error(
				"_ref::exists() not implemented for type: "
				+ component_string <T> ()
			);
		}
	};
public:
	// The get functions will need to be specialized
	template <class T>
	T &get(int i) {
		return _ref <T> ::get(this, i);
	}

	template <class T>
	const T &get(int i) const {
		return _ref <T> ::get(this, i);
	}

	// Existence check
	template <class T>
	bool exists(int i) const {
		return _ref <T> ::exists(this, i);
	}

	// Add a component
	template <class T, class ... Args>
	void add(int i, Args ... args) {
		_ref <T> ::ref(this, i) = _constructor <T> ::make(args ...);
	}

	// Size of ECS
	int size() const {
		return transforms.size();
	}

	// Create a new entity
	Entity make_entity(const std::string &name = "Entity");

	// Display info for one component
	template <class T>
	void info() const {
		std::cout << "Archetype: " << component_string <T> () << std::endl;
		for (size_t i = 0; i < transforms.size(); i++) {
			std::cout << "\tEntity " << i << ": ";
			if (_ref <T> ::exists(this, i))
				std::cout << "yes";
			else
				std::cout << "no";
			std::cout << std::endl;
		}
	}
};

// _constructor specializations
template <>
struct ECS::_constructor <Mesh> {
	template <class ... Args>
	static MeshPtr make(Args ... args) {
		return std::make_shared <Mesh> (args ...);
	}
};

template <>
struct ECS::_constructor <Rasterizer> {
	template <class ... Args>
	static RasterizerPtr make(Args ... args) {
		return std::make_shared <Rasterizer> (args ...);
	}
};

template <>
struct ECS::_constructor <Camera> {
	template <class ... Args>
	static CameraPtr make(Args ... args) {
		return std::make_shared <Camera> (args ...);
	}
};

// _ref specializations
// TODO: another header
template <>
struct ECS::_ref <Transform> {
	static Transform &ref(ECS *ecs, int i) {
		return ecs->transforms[i];
	}

	static Transform &get(ECS *ecs, int i) {
		return ecs->transforms[i];
	}

	static const Transform &get(const ECS *ecs, int i) {
		return ecs->transforms[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return ecs->transforms.size() > i;
	}
};

template <>
struct ECS::_ref <Mesh> {
	static MeshPtr &ref(ECS *ecs, int i) {
		return ecs->meshes[i];
	}

	static Mesh &get(ECS *ecs, int i) {
		return *ecs->meshes[i];
	}

	static const Mesh &get(const ECS *ecs, int i) {
		return *ecs->meshes[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return (ecs->meshes.size() > i)
			&& (ecs->meshes[i] != nullptr);
	}
};

template <>
struct ECS::_ref <Rasterizer> {
	static RasterizerPtr &ref(ECS *ecs, int i) {
		return ecs->rasterizers[i];
	}

	static Rasterizer &get(ECS *ecs, int i) {
		return *ecs->rasterizers[i];
	}

	static const Rasterizer &get(const ECS *ecs, int i) {
		return *ecs->rasterizers[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return (ecs->rasterizers.size() > i)
			&& (ecs->rasterizers[i] != nullptr);
	}
};

template <>
struct ECS::_ref <Camera> {
	static CameraPtr &ref(ECS *ecs, int i) {
		return ecs->cameras[i];
	}

	static Camera &get(ECS *ecs, int i) {
		return *ecs->cameras[i];
	}

	static const Camera &get(const ECS *ecs, int i) {
		return *ecs->cameras[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return (ecs->cameras.size() > i)
			&& (ecs->cameras[i] != nullptr);
	}
};

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
	std::string	name = "";
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
	// Default
	Entity() = default;

	// Non copy, id is unique
	Entity(const Entity &) = delete;
	Entity &operator=(const Entity &) = delete;

	// Moveable
	Entity(Entity &&other)
			: name(std::move(other.name)),
			id(other.id), ecs(other.ecs) {
		other.id = -1;
		other.ecs = nullptr;
	}

	Entity &operator=(Entity &&other) {
		name = std::move(other.name);
		id = other.id;
		ecs = other.ecs;
		other.id = -1;
		other.ecs = nullptr;
		return *this;
	}

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

	// Existence check
	template <class T>
	bool exists() const {
		_assert();
		return ecs->exists <T> (id);
	}

	// Add a component
	template <class T, class ... Args>
	void add(Args ... args) {
		_assert();
		ecs->add <T> (id, args ...);
	}

	// Friend the ECS class
	friend class ECS;
};

}

#endif
