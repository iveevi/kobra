#include "../../include/asmodeus/backend.cuh"

namespace kobra {

namespace asmodeus {

// Update the backend with scene data
void update(Backend &backend, const ECS &ecs)
{
	// Preprocess the entities
	std::vector <const Rasterizer *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	{
		for (int i = 0; i < ecs.size(); i++) {
			// TODO: one unifying renderer component,
			// with options for raytracing, etc
			if (ecs.exists <Rasterizer> (i)) {
				const auto *rasterizer = &ecs.get <Rasterizer> (i);
				const auto *transform = &ecs.get <Transform> (i);

				rasterizers.push_back(rasterizer);
				rasterizer_transforms.push_back(transform);
			}
			
			if (ecs.exists <Light> (i)) {
				const auto *light = &ecs.get <Light> (i);
				const auto *transform = &ecs.get <Transform> (i);

				lights.push_back(light);
				light_transforms.push_back(transform);
			}
		}
	}

	// Check if the objects are dirty
	bool dirty = false;
	for (int i = 0; i < rasterizers.size(); i++) {
		if (i >= backend.scene.c_rasterizers.size()) {
			dirty = true;
			break;
		}

		if (rasterizers[i] != backend.scene.c_rasterizers[i]) {
			dirty = true;
			break;
		}
	}

	// Update the rasterizers
	if (dirty) {
		backend.scene.c_rasterizers = rasterizers;
	}
}

}

}
