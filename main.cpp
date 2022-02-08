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

	// Pixel data
	size_t pixel_size = 4 * sizeof(char) * 800 * 600;
	size_t world_size = sizeof(GPUWorld);
	size_t object_size = sizeof(aligned_vec4) * 16;
	size_t light_size = sizeof(aligned_vec4) * 16;

	VkBufferUsageFlags pixel_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags world_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags object_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags light_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Create buffers
	vulkan.make_buffer(pixel_buffer, pixel_size, pixel_usage);
	vulkan.make_buffer(world_buffer, sizeof(GPUWorld), world_usage);
	vulkan.make_buffer(object_buffer, object_size, object_usage);
	vulkan.make_buffer(light_buffer, light_size, light_usage);

	// vulkan.map_buffer(&world_buffer, &world_data, sizeof(WorldData));
	vulkan.map_buffer(&object_buffer, objects, object_size);
	vulkan.map_buffer(&light_buffer, lights, light_size);

	// Add all buffers to deletion queue
	vulkan.push_deletion_task(
		[&](Vulkan *vk) {
			vk->destroy_buffer(pixel_buffer);
			vk->destroy_buffer(world_buffer);
			vk->destroy_buffer(object_buffer);
			vk->destroy_buffer(light_buffer);
			Logger::ok("[main] Deleted buffers");
		}
	);

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

		/* Update world data
		world_data.camera_position = camera.transform.position;
		world_data.camera_forward = camera.transform.forward;
		world_data.camera_right = camera.transform.right;
		world_data.camera_up = camera.transform.up; */
	
		GPUWorld gworld = world.dump();
		vulkan.map_buffer(&world_buffer, &gworld, sizeof(GPUWorld));

		// Update lights
		float amplitude = 5.0f;
		lights[0] = glm::vec4 {
			amplitude * sin(time), 5.0f,
			amplitude * cos(time), 1.0f
		};

		vulkan.map_buffer(&light_buffer, lights, light_size);

		// Show frame
		vulkan.frame();

		// Time
		time += delta_time;
	}

	vulkan.idle();
}
