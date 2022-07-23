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
#include "lights.hpp"
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
KOBRA_COMPONENT_STRING(Material)
KOBRA_COMPONENT_STRING(Mesh)
KOBRA_COMPONENT_STRING(Rasterizer)
KOBRA_COMPONENT_STRING(Raytracer)
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
	Archetype <MaterialPtr>		materials;
	Archetype <MeshPtr>		meshes;
	Archetype <RasterizerPtr>	rasterizers;
	Archetype <RaytracerPtr>	raytracers;
	Archetype <Transform>		transforms;

	Archetype <Entity>		entities;

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
		// TODO: check index and show warningi with name
		if (!_ref <T>::exists(this, i)) {
			KOBRA_LOG_FUNC(warn) << "Entity " << i << " does not have component "
				<< component_string <T> () << ".\n";
		}

		return _ref <T> ::get(this, i);
	}

	template <class T>
	const T &get(int i) const {
		if (!_ref <T>::exists(this, i)) {
			KOBRA_LOG_FUNC(warn) << "Entity " << i << " does not have component "
				<< component_string <T> () << ".\n";
		}

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
		return entities.size();
	}

	// Get entity
	Entity &get_entity(int i) {
		return entities[i];
	}

	const Entity &get_entity(int i) const {
		return entities[i];
	}

	// Create a new entity
	Entity &make_entity(const std::string &name = "Entity");

	// Display info for one component
	template <class T>
	void info() const;
};

// _constructor specializations
// TODO: macros?
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
struct ECS::_constructor <Raytracer> {
	template <class ... Args>
	static RaytracerPtr make(Args ... args) {
		return std::make_shared <Raytracer> (args ...);
	}
};

template <>
struct ECS::_constructor <Camera> {
	template <class ... Args>
	static CameraPtr make(Args ... args) {
		return std::make_shared <Camera> (args ...);
	}
};

template <>
struct ECS::_constructor <Light> {
	template <class ... Args>
	static LightPtr make(Args ... args) {
		return std::make_shared <Light> (args ...);
	}
};

template <>
struct ECS::_constructor <Material> {
	template <class ... Args>
	static MaterialPtr make(Args ... args) {
		return std::make_shared <Material> (args ...);
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
struct ECS::_ref <Raytracer> {
	static RaytracerPtr &ref(ECS *ecs, int i) {
		return ecs->raytracers[i];
	}

	static Raytracer &get(ECS *ecs, int i) {
		return *ecs->raytracers[i];
	}

	static const Raytracer &get(const ECS *ecs, int i) {
		return *ecs->raytracers[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return (ecs->raytracers.size() > i)
			&& (ecs->raytracers[i] != nullptr);
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

template <>
struct ECS::_ref <Light> {
	static LightPtr &ref(ECS *ecs, int i) {
		return ecs->lights[i];
	}

	static Light &get(ECS *ecs, int i) {
		return *ecs->lights[i];
	}

	static const Light &get(const ECS *ecs, int i) {
		return *ecs->lights[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return (ecs->lights.size() > i)
			&& (ecs->lights[i] != nullptr);
	}
};

template <>
struct ECS::_ref <Material> {
	static MaterialPtr &ref(ECS *ecs, int i) {
		return ecs->materials[i];
	}

	static Material &get(ECS *ecs, int i) {
		return *ecs->materials[i];
	}

	static const Material &get(const ECS *ecs, int i) {
		return *ecs->materials[i];
	}

	static bool exists(const ECS *ecs, int i) {
		return (ecs->materials.size() > i)
			&& (ecs->materials[i] != nullptr);
	}
};

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
