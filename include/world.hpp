#ifndef WORLD_H_
#define WORLD_H_

// Standard headers
#include <memory>
#include <vector>

// Engine headers
#include "camera.hpp"
#include "core.hpp"
#include "light.hpp"
#include "logger.hpp"
#include "primitive.hpp"

// Rendering options
struct Options {
	bool	debug_bvh = false;
	int	discretize = -1;
};

// GPU friendyl world data structure
struct GPUWorld {
	// GLSL-like aliases
	using uint = uint32_t;

	// Data
	uint	objects;
	uint	primitives;
	uint	lights;

	uint	width;
	uint	height;

	uint	options;

	int32_t	discretize;

	// Camera data
	aligned_vec4 position;
	aligned_vec4 forward;
	aligned_vec4 up;
	aligned_vec4 right;

	aligned_vec4 tunings;
};

// API friendly world structure
struct World {
	using PrimitivePtr = std::shared_ptr <Primitive>;
	using LightPtr = std::shared_ptr <Light>;

	// Data
	Camera				camera;
	std::vector <PrimitivePtr>	objects;
	std::vector <LightPtr>		lights;

	// Extra
	Options				options;

	// World constructor
	World() {}
	World(const Camera &camera,
			const std::vector <PrimitivePtr> &objects,
			const std::vector <LightPtr> &lights)
			: camera(camera), objects(objects),
			lights(lights) {}

	// Dump data to GPU friendly structure
	GPUWorld dump() const {
		GPUWorld world {
			// TODO: modify object count function
			.objects = static_cast <uint> (objects.size()),
			.primitives = 0,
			.lights = static_cast <uint> (lights.size()),
			.width = 800,
			.height = 600,
			.discretize = options.discretize
		};

		// Add primitives
		for (const auto &object : objects)
			world.primitives += object->count();

		// Camera data
		world.position = camera.transform.position;
		world.forward = camera.transform.forward;
		world.up = camera.transform.up;
		world.right = camera.transform.right;

		/* world.fov = camera.tunings.fov;
		world.scale = camera.tunings.scale;
		world.aspect = camera.tunings.aspect; */
		world.tunings = glm::vec3 {
			camera.tunings.fov,
			camera.tunings.scale,
			camera.tunings.aspect
		};

		// Set options as bit flags
		world.options = 0;
		if (options.debug_bvh)
			world.options |= 0x1;

		return world;
	}

	// Write object data to buffer
	void write_objects(Buffer &buffer, Buffer &materials, Indices &indices) const {
		buffer.clear();
		materials.clear();

		indices.push_back(0);
		for (const auto &object : objects) {
			// uint index = materials.size();
			// object->material.write_to_buffer(materials);
			object->write_to_buffer(buffer, materials, indices);
			indices.push_back(buffer.size());
		}

		// Pop last index
		indices.pop_back();
	}

	// Write light data to buffer
	void write_lights(Buffer &buffer, Indices &indices) const {
		buffer.clear();

		indices.push_back(0);
		for (const auto &light : lights) {
			light->write_to_buffer(buffer);
			indices.push_back(buffer.size());
		}

		// Pop last index
		indices.pop_back();
	}

	// Extract all bounding bxoes from the primitives
	std::vector <mercury::BoundingBox> extract_bboxes() const {
		std::vector <mercury::BoundingBox> bboxes;
		for (const auto &object : objects)
			object->extract_bboxes(bboxes);
		return bboxes;
	}
};

#endif
