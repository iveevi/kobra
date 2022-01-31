#ifndef WORLD_H_
#define WORLD_H_

// Standard headers
#include <vector>

// Engine headers
#include "camera.hpp"
#include "light.hpp"
#include "object.hpp"

// World structure for holding all the data
struct World {
	// Camera
	Camera camera;

	// Lights
	std::vector <Light *> lights;

	// Objects
	std::vector <Renderable *> objects;

	// Destructor
	~World() {
		// Delete all lights
		for (unsigned int i = 0; i < lights.size(); i++)
			delete lights[i];

		// Delete all renderables
		for (unsigned int i = 0; i < objects.size(); i++)
			delete objects[i];
	}
};

#endif
