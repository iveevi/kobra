#include "../include/ecs.hpp"

namespace kobra {

// Creating a new entity
Entity &ECS::make_entity(const std::string &name) {
	_expand_all();
	int32_t id = transforms.size() - 1;

	Entity e(name, id, this);
	entities.push_back(e);
	return entities.back();
}

// Private helpers
void ECS::_expand_all()
{
	cameras.push_back(nullptr);
	lights.push_back(nullptr);
	meshes.push_back(nullptr);
	rasterizers.push_back(nullptr);
	transforms.push_back(Transform());

	// TODO: assert that all arrays are the same size
}

}
