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

// GPU friendyl world data structure
struct GPUWorld {
	// GLSL-like aliases
	using uint = uint32_t;

	// Data
	uint	objects;
	uint	lights;

	uint	width;
	uint	height;

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
	Camera			camera;
	std::vector <PrimitivePtr>	objects;
	std::vector <LightPtr>	lights;

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
			.objects = static_cast <uint> (objects.size()),
			.lights = static_cast <uint> (lights.size()),
			.width = 800,
			.height = 600
		};

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