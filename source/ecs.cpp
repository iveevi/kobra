#include "../include/ecs.hpp"

// Macro for get specializations
#define ECS_GET_IMPLEMENTATION(T, array) \
	template <>				\
	T &ECS::get <T> (int i) {		\
		return array[i];		\
	}					\
						\
	template <>				\
	const T &ECS::get <T> (int i) const {	\
		return array[i];		\
	}

namespace kobra {

// Get specializations
ECS_GET_IMPLEMENTATION(Transform, transforms)

// Creating a new entity
Entity ECS::make_entity(const std::string &name) {
	_expand_all();
	int32_t id = transforms.size() - 1;
	return Entity(name, id, this);
}

// Private helpers
void ECS::_expand_all()
{
	transforms.push_back(Transform());

	// TODO: assert that all arrays are the same size
}

}
