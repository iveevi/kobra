// Standard headers
#include <iostream>
#include <cstring>

// Local headers
#include "global.hpp"

// Global camera
Camera camera {
        Transform {
		glm::vec3(0.0f, 0.0f, -4.0f)
	},
 
        Tunings {
                45.0f,
                800,
                600
        }
};

// Aligned structures
struct alignas(16) aligned_vec3 {
	glm::vec3 data;

	aligned_vec3() {}
	aligned_vec3(const glm::vec3 &d) : data(d) {}
};

struct alignas(16) aligned_vec4 {
	glm::vec4 data;

	aligned_vec4() {}
	aligned_vec4(const glm::vec4 &d) : data(d) {}
};

// Buffer type aliases
using Buffer = std::vector <aligned_vec4>;

// Object structures
struct Object {
	float		id = OBJECT_TYPES[OBJT_NONE];
	Transform	transform;

	// Object constructors
	Object() {}
	Object(float id, const Transform &transform)
			: id(id), transform(transform) {}

	// Virtual object destructor
	virtual ~Object() {}

	// Write data to aligned_vec4 buffer (inherited)
	virtual void write(Buffer &buffer) const = 0;

	// Write full object data
	void write_to_buffer(Buffer &buffer) {
		// Push ID, then everythig else
		buffer.push_back(aligned_vec4 {
			glm::vec4(id, 0.0, 0.0, 0.0)
		});

		this->write(buffer);
	}
};

// Sphere structure
struct Sphere : Object {
	float		radius;

	Sphere() {}
	Sphere(float radius, const Transform &transform)
			: Object(OBJECT_TYPES[OBJT_SPHERE], transform),
			radius(radius) {}

	void write(Buffer &buffer) const override {
		buffer.push_back(aligned_vec4 {
			glm::vec4(transform.position, radius)
		});
	}
};

// Plane structure
struct Plane : Object {
	float		length;
	float		width;

	Plane() {}
	Plane(float length, float width, const Transform &transform)
			: Object(OBJECT_TYPES[OBJT_PLANE], transform),
			length(length), width(width) {}

	void write(Buffer &buffer) const override {
		// Position
		buffer.push_back(aligned_vec4 {
			glm::vec4(transform.position, 1.0f)
		});

		// Right, length
		buffer.push_back(aligned_vec4 {
			glm::vec4(transform.right, length)
		});

		// Forward, width
		buffer.push_back(aligned_vec4 {
			glm::vec4(transform.forward, width)
		});
	}
};

// Light structure
struct Light {
	Transform	transform;
	float		intensity;

	// Light constructors
	Light() {}
	Light(const Transform &transform, float intensity)
			: transform(transform), intensity(intensity) {}

	// Virtual light destructor
	virtual ~Light() {}

	// Write data to aligned_vec4 buffer
	virtual void write(Buffer &buffer) const = 0;
};

// API friendly world structure
struct World {

};

// World structur
// TODO: refactor to GPU world,
// store cpu world in another structure that is called World
struct WorldData {
	uint32_t objects;
	uint32_t lights;
	uint32_t backgound;

	uint32_t width = 800;
	uint32_t height = 600;

	aligned_vec3 camera_position = glm::vec3(0.0f, 0.0f, -15.0f);
	alignas(16) glm::vec3 camera_forward = glm::vec3(0.0f, 0.0f, 1.0f);
	alignas(16) glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
	alignas(16) glm::vec3 camera_right = glm::vec3(1.0f, 0.0f, 0.0f);

	float fov = camera.tunings.fov;
	float scale = camera.tunings.scale;
	float aspect = camera.tunings.aspect;

	// TODO: store indices for objects and lights
};

WorldData world_data = {
	.objects = 6,
	.lights = 1,
	.backgound = 0x202020,
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
	size_t world_size = sizeof(WorldData);
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
	vulkan.make_buffer(world_buffer, sizeof(WorldData), world_usage);
	vulkan.make_buffer(object_buffer, object_size, object_usage);
	vulkan.make_buffer(light_buffer, light_size, light_usage);

	vulkan.map_buffer(&world_buffer, &world_data, sizeof(WorldData));
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

		// Update world data
		world_data.camera_position = camera.transform.position;
		world_data.camera_forward = camera.transform.forward;
		world_data.camera_right = camera.transform.right;
		world_data.camera_up = camera.transform.up;
		vulkan.map_buffer(&world_buffer, &world_data, sizeof(WorldData));

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
