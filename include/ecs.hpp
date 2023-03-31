#ifndef KOBRA_ECS_H_
#define KOBRA_ECS_H_

// Standard headers
#include <memory>
#include <unordered_map>
#include <vector>

// GLM headers
#include <glm/gtx/string_cast.hpp>

// Engine headers
#include "camera.hpp"
#include "common.hpp"
#include "lights.hpp"
#include "logger.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "renderable.hpp"
#include "transform.hpp"

namespace kobra {

// Forward declarations
class Entity;
class Scene;

// Component to string
template <typename T>
std::string component_string()
{
	return "<Undefined>";
}

// Specilizations
// TODO: macrofy
#define KOBRA_COMPONENT_STRING(T)			\
	template <>					\
	inline std::string component_string <T> ()	\
	{						\
		return #T;				\
	}

KOBRA_COMPONENT_STRING(Camera)
KOBRA_COMPONENT_STRING(Light)
KOBRA_COMPONENT_STRING(Mesh)
KOBRA_COMPONENT_STRING(Renderable)
KOBRA_COMPONENT_STRING(Transform)

// Components which all entities must have
// are stored by value
//
// The others are stored as pointers
template <class T>
using Archetype = std::vector <T>;

class ECS {
	Archetype <CameraPtr>		cameras;
	Archetype <LightPtr>		lights;
	Archetype <MeshPtr>		meshes;
	Archetype <RenderablePtr>	rasterizers;
	Archetype <Transform>		transforms;

	Archetype <Entity>		entities;

	std::unordered_map <std::string, int>
					name_map;

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
		using Return = void;

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
		// TODO: check index and show warningi with name
		if (!_ref <T>::exists(this, i)) {
			KOBRA_LOG_FUNC(Log::WARN) << "Entity " << i << " does not have component "
				<< component_string <T> () << ".\n";
		}

		return _ref <T> ::get(this, i);
	}

	template <class T>
	const T &get(int i) const {
		if (!_ref <T>::exists(this, i)) {
			KOBRA_LOG_FUNC(Log::WARN) << "Entity " << i << " does not have component "
				<< component_string <T> () << ".\n";
		}

		return _ref <T> ::get(this, i);
	}

	// Multiple existence check
	template <class T, class ... Ts>
	bool exists(int i) const {
		if constexpr (sizeof ... (Ts) == 0)
			return _ref <T> ::exists(this, i);
		else
			return _ref <T> ::exists(this, i) && exists <Ts...> (i);
	}

	// Extract tuples of components from all entities
	template <class ... Ts>
	std::vector <std::tuple <const Ts *...>> tuples() const {
		std::vector <std::tuple <const Ts *...>> ret;
		for (int i = 0; i < size(); i++) {
			if (exists <Ts...> (i))
				ret.push_back(std::make_tuple(&get <Ts> (i) ...));
		}

		return ret;
	}

	// Add a component
	template <class T, class ... Args>
	void add(int i, Args ... args) {
		_ref <T> ::ref(this, i) = _constructor <T> ::make(args ...);
	}

	// Size of ECS
	int size() const {
		return entities.size();
	}

	// Get entity
	Entity &get_entity(int i) {
		return entities[i];
	}

	const Entity &get_entity(int i) const {
		return entities[i];
	}

	// Get entity by name
	Entity &get_entity(const std::string &name) {
		if (name_map.find(name) == name_map.end())
			KOBRA_LOG_FUNC(Log::WARN) << "Entity " << name << " does not exist.\n";

		return entities[name_map.at(name)];
	}

	const Entity &get_entity(const std::string &name) const {
		if (name_map.find(name) == name_map.end())
			KOBRA_LOG_FUNC(Log::WARN) << "Entity " << name << " does not exist.\n";
		return entities[name_map.at(name)];
	}

	// Create a new entity
	Entity &make_entity(const std::string &name = "Entity");

	// Iterators
	Archetype <Entity>::iterator begin() {
		return entities.begin();
	}

	Archetype <Entity>::iterator end() {
		return entities.end();
	}

	Archetype <Entity>::const_iterator begin() const {
		return entities.begin();
	}

	Archetype <Entity>::const_iterator end() const {
		return entities.end();
	}

	// Display info for one component
	template <class T>
	void info() const;

	// Methods for saving and loading
	void populate_mesh_cache(std::set <const Submesh *> &cache) const {
		for (const MeshPtr &mesh : meshes) {
			if (mesh) {
				std::cout << "Mesh with " << mesh->triangles() << " triangles\n";
				mesh->populate_mesh_cache(cache);
			}
		}
	}
};

// _constructor specializations
// TODO: macros?
#define KOBRA_MAKE_SHARED(T, Ret)				\
	template <>						\
	struct ECS::_constructor <T> {				\
		template <class ... Args>			\
		static Ret make(Args ... args) {		\
			return std::make_shared <T> (args...);	\
		}						\
	}

KOBRA_MAKE_SHARED(Camera, CameraPtr);
KOBRA_MAKE_SHARED(Light, LightPtr);
KOBRA_MAKE_SHARED(Mesh, MeshPtr);
KOBRA_MAKE_SHARED(Renderable, RenderablePtr);

// _ref specializations
// TODO: another header
#define KOBRA_REF(T, Array)					\
	template <>						\
	struct ECS::_ref <T> {					\
		using Return = T;				\
								\
		static T &ref(ECS *ecs, int i) {		\
			return ecs->Array[i];			\
		}						\
								\
		static T &get(ECS *ecs, int i) {		\
			return ecs->Array[i];			\
		}						\
								\
		static const T &get(const ECS *ecs, int i) {	\
			return ecs->Array[i];			\
		}						\
		static bool exists(const ECS *ecs, int i) {	\
			return ecs->Array.size() > i;		\
		}						\
	}

KOBRA_REF(Transform, transforms);

// TODO: we shouldnt need to use refs...
// if it is large, then move it...
#define KOBRA_RET_SHARED(T, Ret, Array)				\
	template <>						\
	struct ECS::_ref <T> {					\
		using Return = Ret;				\
								\
		static Ret &ref(ECS *ecs, int i) {		\
			return ecs->Array[i];			\
		}						\
								\
		static T &get(ECS *ecs, int i) {		\
			return *ecs->Array[i];			\
		}						\
								\
		static const T &get(const ECS *ecs, int i) {	\
			return *ecs->Array[i];			\
		}						\
		static bool exists(const ECS *ecs, int i) {	\
			return (ecs->Array.size() > i)		\
				&& (ecs->Array[i] != nullptr);	\
		}						\
	}

KOBRA_RET_SHARED(Camera, CameraPtr, cameras);
KOBRA_RET_SHARED(Light, LightPtr, lights);
KOBRA_RET_SHARED(Mesh, MeshPtr, meshes);
KOBRA_RET_SHARED(Renderable, RenderablePtr, rasterizers);

// Entity class, acts like a pointer to a component
class Entity {
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
	std::string	name = "";

	// Default
	Entity() = default;

	// Copy
	Entity(const Entity &) = default;
	Entity &operator=(const Entity &) = default;

	// Unmoveable
	Entity(Entity &&) = delete;
	Entity &operator=(Entity &&) = delete;

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
	template <class ... Ts>
	bool exists() const {
		_assert();
		return ecs->exists <Ts...> (id);
	}

	// Add a component
	template <class T, class ... Args>
	void add(Args ... args) {
		_assert();
		ecs->add <T> (id, args ...);
	}

	// Friend the ECS class
	friend class ECS;
	friend class Scene;
};

template <class T>
void ECS::info() const
{
	std::cout << "Archetype: " << component_string <T> () << std::endl;
	for (size_t i = 0; i < size(); i++) {
		std::cout << "\tEntity " << entities[i].name << ": ";
		if (_ref <T> ::exists(this, i))
			std::cout << "yes";
		else
			std::cout << "no";
		std::cout << std::endl;
	}
}

// Specializations of info
template <>
inline void ECS::info <Transform> () const
{
	std::cout << "Archetype: " << component_string <Transform> () << std::endl;
	for (size_t i = 0; i < transforms.size(); i++) {
		std::cout << "\tEntity " << i << ": .pos = "
			<< glm::to_string(transforms[i].position) << std::endl;
	}
}

}

#endif
