#include "include/system.hpp"

namespace kobra {

// Creating a new entity
Entity &System::make_entity(const std::string &name) {
	_expand_all();
        // TODO: when allowing removel of System objects, create a more complex
        // id...
	int32_t id = transforms.size() - 1;

	Entity e(name, id, this);
	entities.push_back(e);

	lookup[name] = id;
        printf("System refs:\n");
        for (int i = 0; i < transforms.size(); i++) {
                std::string name = "";
                for (auto pr : lookup) {
                        if (pr.second == i)
                                name = pr.first;
                }
                printf("%s -- tr: %p renderable: %p\n", name.c_str(), &transforms[i], &rasterizers[i]);
        }
	return entities.back();
}

// Private helpers
void System::_expand_all()
{
	cameras.push_back(nullptr);
	lights.push_back(nullptr);
	meshes.push_back(nullptr);
	rasterizers.push_back(nullptr);
	transforms.push_back(Transform());

	// TODO: assert that all arrays are the same size
}

}
