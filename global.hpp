#ifndef GLOBAL_H_
#define GLOBAL_H_

#define MERCURY_VALIDATION_LAYERS
#define MERCURY_VALIDATION_ERROR_ONLY
// #define MERCURY_THROW_ERROR

// Standard headers
#include <memory>

// Engine headers
#include "include/backend.hpp"
#include "include/camera.hpp"
#include "include/core.hpp"
#include "include/logger.hpp"
#include "include/mesh.hpp"
#include "include/model.hpp"
#include "include/primitive.hpp"
#include "include/types.h"

// Light structure
struct Light {
	float		id = LIGHT_TYPE_POINT;
	Transform	transform;
	float		intensity;

	// Light constructors
	Light() {}
	Light(float x, const Transform &t, float i)
			: id(x), transform(t), intensity(i) {}

	// Virtual light destructor
	virtual ~Light() {}

	// Write data to aligned_vec4 buffer
	virtual void write(Buffer &buffer) const = 0;

	// Write full light data
	void write_to_buffer(Buffer &buffer) {
		// Push ID, then everythig else
		buffer.push_back(aligned_vec4 {
			glm::vec4 {
				id, transform.position.x,
				transform.position.y,
				transform.position.z
			}
		});

		this->write(buffer);
	}
};

// Point light
struct PointLight : public Light {
	PointLight() {}
	PointLight(const Transform &t, float i)
			: Light(LIGHT_TYPE_POINT, t, i) {}

	void write(Buffer &buffer) const override {}
};

// GPU friendyl world data structure
// TODO: header
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

	/* float fov;
	float scale;
	float aspect; */
	aligned_vec4 tunings;
};

// API friendly world structure
// TODO: header
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
};

// Pixel buffers
extern Vulkan::Buffer pixel_buffer;
extern Vulkan::Buffer world_buffer;
extern Vulkan::Buffer objects_buffer;
extern Vulkan::Buffer lights_buffer;
extern Vulkan::Buffer materials_buffer;

// Compute shader
extern VkShaderModule compute_shader;

// Global world data
// extern Camera camera;
extern World world;

// Vulkan/GLFW helpers
void cmd_buffer_maker(Vulkan *, size_t);
void descriptor_set_maker(Vulkan *, size_t);

void key_callback(GLFWwindow *, int, int, int, int);

void mouse_callback(GLFWwindow *, double, double);

#endif
