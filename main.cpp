// Standard headers
#include <cstring>
#include <iostream>

// Local headers
#include "global.hpp"

// List of objet materials
Material materials[] {
	glm::vec3 {0.1f, 0.5f, 0.2f},
	glm::vec3 {0.9f, 0.5f, 0.2f},
	glm::vec3 {0.4f, 0.4f, 0.9f},
	glm::vec3 {0.5f, 0.1f, 0.6f},
	glm::vec3 {0.6f, 0.5f, 0.3f},
	glm::vec3 {1.0f, 0.5f, 1.0f}
};

// List of object transforms
Transform transforms[] {
	glm::vec3 {-1.0f, 0.0f, 4.0f},
	glm::vec3 {0.5f, 5.0f, 3.0f},
	glm::vec3 {6.0f, -2.0f, 5.0f},
	glm::vec3 {6.0f, 3.0f, 11.5f},
	glm::vec3 {6.0f, 3.0f, -2.0f},
	glm::vec3 {0.0f, 0.0f, 0.0f},
	glm::vec3 {0.0f, 0.0f, 0.0f}
};
		
World world {
	// Camera
	Camera {
		Transform {
			glm::vec3(0.0f, 0.0f, -4.0f)
		},
	 
		Tunings {
			45.0f, 800, 600
		}
	},

	// Objects
	// TODO: later read from file
	std::vector <World::ObjectPtr> {
		World::ObjectPtr(new Sphere(1.0f, transforms[0], materials[0])),
		World::ObjectPtr(new Sphere(3.0f, transforms[1], materials[1])),
		World::ObjectPtr(new Sphere(6.0f, transforms[2], materials[2])),
		World::ObjectPtr(new Sphere(2.0f, transforms[3], materials[3])),
		World::ObjectPtr(new Sphere(2.0f, transforms[4], materials[4])),
		World::ObjectPtr(new Sphere(0.2f, transforms[5], materials[5]))
	},

	// Lights
	std::vector <World::LightPtr> {
		World::LightPtr(new PointLight(transforms[6], 0.0f))
	}
};

// Print aligned_vec4
// TODO: common header
inline std::ostream& operator<<(std::ostream& os, const glm::vec4 &v)
{
	return (os << "(" << v.x << ", " << v.y
		<< ", " << v.z << ", " << v.w << ")");
}

inline std::ostream &operator<<(std::ostream &os, const aligned_vec4 &v)
{
	return (os << v.data);
}

// TODO: put the following functions into a alloc.cpp file

// Minimum sizes
#define INITIAL_OBJECTS	100UL
#define INITIAL_LIGHTS	100UL

// Size of the world data, including indices
size_t world_data_size()
{
	size_t objects = std::max(world.objects.size(), INITIAL_OBJECTS);
	size_t lights = std::max(world.lights.size(), INITIAL_LIGHTS);

	return sizeof(GPUWorld) + 4 * (objects + lights);
}

// Copy buffer helper
std::pair <uint8_t *, size_t> map_world_buffer(Buffer &objects, Buffer &lights)
{
	// Static (cached) raw memory buffer
	static uint8_t *buffer = nullptr;
	static size_t buffer_size = 0;

	// Check buffer resize requirement
	size_t size = world_data_size();

	// Resize buffer if required
	if (size > buffer_size) {
		buffer_size = size;
		buffer = (uint8_t *) realloc(buffer, buffer_size);
	}

	// Generate world data and write to buffers
	Indices indices;
	world.write_objects(objects, indices);
	world.write_lights(lights, indices);

	Logger::ok() << "Objects indices: ";
	for (auto i : indices)
		Logger::plain() << i << " ";
	Logger::plain() << std::endl;

	// Copy world and indices
	GPUWorld gworld = world.dump();
	memcpy(buffer, &gworld, sizeof(GPUWorld));
	memcpy(buffer + sizeof(GPUWorld), indices.data(),
		4 * indices.size());

	// Return pointer to the buffer
	return {buffer, buffer_size};
}

// Map all the buffers
// TODO: deal with resizing buffers
void map_buffers(Vulkan *vk)
{
	// Create and write to buffers
	Buffer objects;
	Buffer lights;

	auto wb = map_world_buffer(objects, lights);
	Logger::ok() << "World data size: " << wb.second << std::endl;

	// Map buffers
	vk->map_buffer(&world_buffer, wb.first, wb.second);
	vk->map_buffer(&objects_buffer, objects.data(), objects.size());
	vk->map_buffer(&lights_buffer, lights.data(), lights.size());
}

// Allocate buffers
void allocate_buffers(Vulkan &vulkan)
{
	// Sizes of objects and lights
	// are assumed to be the maximum
	static const size_t MAX_OBJECT_SIZE = sizeof(Sphere);
	static const size_t MAX_LIGHT_SIZE = sizeof(PointLight);

	// Allocate buffers
	size_t pixel_size = 4 * 800 * 600;
	size_t world_size = world_data_size();
	size_t objects_size = MAX_OBJECT_SIZE * std::max(world.objects.size(), INITIAL_OBJECTS);
	size_t lights_size = MAX_LIGHT_SIZE * std::max(world.lights.size(), INITIAL_LIGHTS);

	static const VkBufferUsageFlags buffer_usage =
		VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Create buffers
	vulkan.make_buffer(pixel_buffer,   pixel_size,   buffer_usage);
	vulkan.make_buffer(world_buffer,   world_size,   buffer_usage);
	vulkan.make_buffer(objects_buffer, objects_size, buffer_usage);
	vulkan.make_buffer(lights_buffer,  lights_size,  buffer_usage);
	
	// Add all buffers to deletion queue
	vulkan.push_deletion_task(
		[&](Vulkan *vk) {
			vk->destroy_buffer(pixel_buffer);
			vk->destroy_buffer(world_buffer);
			vk->destroy_buffer(objects_buffer);
			vk->destroy_buffer(lights_buffer);
			Logger::ok("[main] Deleted buffers");
		}
	);
}

int main()
{
	// Initialize Vulkan
	Vulkan vulkan;

	// Initialize objects
	aligned_vec4 *objects = new aligned_vec4[16] {
		{ {-1.0f, 0.0f, 4.0f, 1.0f} },
		{ {0.1f, 0.5f, 0.2f, 0.0f} },
		
		{ {0.5f, 5.0f, 3.0f, 3.0f } },
		{ {0.9f, 0.5f, 0.2f, 0.0f} },
		
		{ {6.0f, -2.0f, 5.0f, 6.0f } },
		{ {0.4f, 0.4f, 0.9f, 0.0f} },
		
		{ {6.0f, 3.0f, 11.5f, 2.0f} },
		{ {0.5f, 0.1f, 0.6f, 0.0f} },
		
		{ {6.0f, 3.0f, -2.0f, 2.0f} },
		{ {0.6f, 0.5f, 0.3f, 0.0f} },

		{ {0.0f, 0.0f, 0.0f, 0.2f} },
		{ {1.0f, 0.5f, 1.0f, 1.0f} }
	};

	aligned_vec4 *lights = new aligned_vec4[16] {
		{ {0.0f, 0.0f, 0.0f, 1.0f} }
	};

	/* Pixel data
	size_t pixel_size = 4 * sizeof(char) * 800 * 600;
	size_t world_size = sizeof(GPUWorld);
	size_t object_size = sizeof(aligned_vec4) * 16;
	size_t light_size = sizeof(aligned_vec4) * 16; */

	allocate_buffers(vulkan);

	// Compute shader descriptor
	compute_shader = vulkan.make_shader("shaders/pixel.spv");

	// Add shader to deletion queue'
	vulkan.push_deletion_task(
		[&](Vulkan *vk) {
			vkDestroyShaderModule(vk->device, compute_shader, nullptr);
			Logger::ok("[main] Deleted compute shader");
		}
	);

	// Buffer descriptor
	for (size_t i = 0; i < vulkan.swch_images.size(); i++)
		descriptor_set_maker(&vulkan, i);

	// Set keyboard callback
	glfwSetKeyCallback(vulkan.window, key_callback);

	// Mouse options
	glfwSetInputMode(vulkan.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set mouse callback
	glfwSetCursorPosCallback(vulkan.window, mouse_callback);
	
	vulkan.set_command_buffers(cmd_buffer_maker);

	// ImGui and IO
	vulkan.init_imgui();
	ImGuiIO &io = ImGui::GetIO();

	// Main render loop
	float time = 0.0f;
	float delta_time = 0.01f;
	while (!glfwWindowShouldClose(vulkan.window)) {
		glfwPollEvents();
	
		/* GPUWorld gworld = world.dump();
		vulkan.map_buffer(&world_buffer, &gworld, sizeof(GPUWorld));

		// Update lights
		float amplitude = 5.0f;
		lights[0] = glm::vec4 {
			amplitude * sin(time), 5.0f,
			amplitude * cos(time), 1.0f
		};

		vulkan.map_buffer(&lights_buffer, lights, light_size); */
		
		float amplitude = 5.0f;
		glm::vec3 position {
			amplitude * sin(time), 5.0f,
			amplitude * cos(time)
		};

		world.lights[0]->transform.position = position;

		// Update buffers
		map_buffers(&vulkan);

		// Show frame
		vulkan.frame();

		// Time
		time += delta_time;
	}

	vulkan.idle();
}
