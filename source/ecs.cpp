#include "../include/ecs.hpp"

namespace kobra {

// Creating a new entity
Entity &ECS::make_entity(const std::string &name) {
	_expand_all();
        // TODO: when allowing removel of ECS objects, create a more complex
        // id...
	int32_t id = transforms.size() - 1;

	Entity e(name, id, this);
	entities.push_back(e);

	name_map[name] = id;
        printf("ECS refs:\n");
        for (int i = 0; i < transforms.size(); i++) {
                std::string name = "";
                for (auto pr : name_map) {
                        if (pr.second == i)
                                name = pr.first;
                }
                printf("%s -- tr: %p renderable: %p\n", name.c_str(), &transforms[i], &rasterizers[i]);
        }
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
