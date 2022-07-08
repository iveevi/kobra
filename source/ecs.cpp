#include "../include/ecs.hpp"

namespace kobra {

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
	meshes.push_back(nullptr);

	// TODO: assert that all arrays are the same size
}

}
