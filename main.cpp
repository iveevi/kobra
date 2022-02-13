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
	glm::vec3 {1.0f, 0.5f, 1.0f},
	{glm::vec3 {1.0f, 1.0f, 1.0f}, SHADING_TYPE_LIGHT},
};

// List of object transforms
Transform transforms[] {
	glm::vec3 {-1.0f, 0.0f, 4.0f},
	glm::vec3 {0.5f, 5.0f, 3.0f},
	glm::vec3 {6.0f, -2.0f, 5.0f},
	glm::vec3 {6.0f, 3.0f, 11.5f},
	glm::vec3 {6.0f, 3.0f, -2.0f},
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

	// Primitives
	// TODO: later read from file
	std::vector <World::PrimitivePtr> {
		World::PrimitivePtr(new Sphere(0.25f, transforms[0], materials[6])),
		World::PrimitivePtr(new Sphere(1.0f, transforms[0], materials[0])),
		World::PrimitivePtr(new Sphere(3.0f, transforms[1], materials[1])),
		World::PrimitivePtr(new Sphere(6.0f, transforms[2], materials[2])),
		World::PrimitivePtr(new Sphere(2.0f, transforms[3], materials[3])),
		World::PrimitivePtr(new Sphere(2.0f, transforms[4], materials[4])),
		World::PrimitivePtr(new Triangle(
			glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f),
			glm::vec3(1.0f, 0.0f, 0.0f),
			materials[5]
		)),
		// Cube mesh
		World::PrimitivePtr(new Mesh <VERTEX_TYPE_POSITION> (
			{
				glm::vec3(0.0f, 0.0f, -1.0f),
				glm::vec3(1.0f, 0.0f, -1.0f),
				glm::vec3(1.0f, 1.0f, -1.0f),
				glm::vec3(0.0f, 1.0f, -1.0),
				glm::vec3(0.0f, 0.0f, 1.0f),
				glm::vec3(1.0f, 0.0f, 1.0f),
				glm::vec3(1.0f, 1.0f, 1.0f),
				glm::vec3(0.0f, 1.0f, 1.0f)
			},
			{
				0, 1, 2,
				0, 2, 3,
				4, 5, 6,
				4, 6, 7,
				0, 4, 7,
				0, 7, 3,
				1, 5, 6,
				1, 6, 2,
				0, 1, 5,
			},
			materials[1]
		))
	},

	// Lights
	std::vector <World::LightPtr> {
		World::LightPtr(new PointLight(transforms[0], 0.0f))
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
#define INITIAL_OBJECTS		100UL
#define INITIAL_LIGHTS		100UL
#define INITIAL_MATERIALS	100UL

// Sizes of objects and lights
// are assumed to be the maximum
static const size_t MAX_OBJECT_SIZE = sizeof(Triangle);
static const size_t MAX_LIGHT_SIZE = sizeof(PointLight);


// Size of the world data, including indices
size_t world_data_size()
{
	size_t objects = std::max(world.objects.size(), INITIAL_OBJECTS);
	size_t lights = std::max(world.lights.size(), INITIAL_LIGHTS);

	return sizeof(GPUWorld) + 4 * (objects + lights);
}
// Copy buffer helper
std::pair <uint8_t *, size_t> map_world_buffer(Vulkan *vk, Buffer &objects, Buffer &lights, Buffer &materials)
{
	// Static (cached) raw memory buffer
	static uint8_t *buffer = nullptr;
	static size_t buffer_size = 0;
	
	static const VkBufferUsageFlags buffer_usage =
		VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Check buffer resize requirement
	size_t size = world_data_size();

	// Resize buffer if required
	if (size > buffer_size) {
		buffer_size = size;
		buffer = (uint8_t *) realloc(buffer, buffer_size);
	}

	// Generate world data and write to buffers
	Indices indices;
	world.write_objects(objects, materials, indices);
	world.write_lights(lights, indices);

	// Copy world and indices
	GPUWorld gworld = world.dump();
	gworld.objects = objects.size();

	memcpy(buffer, &gworld, sizeof(GPUWorld));
	memcpy(buffer + sizeof(GPUWorld), indices.data(),
		4 * indices.size());

	/* Dump contents of the buffers
	// TODO: ImGui option
	std::cout << "=== Objects: " << objects.size() << " ===" << std::endl;
	for (size_t i = 0; i < objects.size(); i++)
		std::cout << i << ":\t" << objects[i] << std::endl;
	std::cout << "Lights: " << lights.size() << std::endl;
	for (size_t i = 0; i < lights.size(); i++)
		std::cout << lights[i] << std::endl;
	std::cout << "Materials: " << materials.size() << std::endl;
	for (size_t i = 0; i < materials.size(); i++)
		std::cout << materials[i] << std::endl;
	std::cout << "Indices: " << indices.size() << std::endl;
	for (size_t i = 0; i < indices.size(); i++)
		std::cout << indices[i] << std::endl;

	// Dump buffer contents (as uvec4)
	std::cout << "=== Buffer contents ===" << std::endl;
	for (size_t i = 0; i < buffer_size; i += 4 * sizeof(uint32_t)) {
		uint32_t *uptr = (uint32_t *) (buffer + i);
		float *fptr = (float *) (buffer + i);

		std::cout << uptr[0] << " " << uptr[1] << " "
			<< uptr[2] << " " << uptr[3]
			<< " -> ("
			<< fptr[0] << ", " << fptr[1] << ", "
			<< fptr[2] << ", " << fptr[3]
			<< ")" << std::endl;
	} */
	
	// Resizing and remaking buffers
	int resized = 0;

	// World
	if (size > world_buffer.size) {
		// Resize vulkan buffer
		vk->destroy_buffer(world_buffer);
		vk->make_buffer(world_buffer, buffer_size, buffer_usage);
		resized++;
	}

	// Objects
	if (sizeof(aligned_vec4) * objects.size() > objects_buffer.size) {
		// Resize vulkan buffer
		vk->destroy_buffer(objects_buffer);
		vk->make_buffer(objects_buffer, objects.size() * sizeof(aligned_vec4), buffer_usage);
		resized++;
	}

	// Lights
	if (sizeof(aligned_vec4) * lights.size() > lights_buffer.size) {
		// Resize vulkan buffer
		vk->destroy_buffer(lights_buffer);
		vk->make_buffer(lights_buffer, lights.size() * sizeof(aligned_vec4), buffer_usage);
		resized++;
	}

	// Update descriptor sets
	if (resized) {
		for (size_t i = 0; i < vk->swch_images.size(); i++)
			descriptor_set_maker(vk, i);
		vk->set_command_buffers(cmd_buffer_maker);
	}

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
	Buffer materials;

	auto wb = map_world_buffer(vk, objects, lights, materials);

	// Map buffers
	vk->map_buffer(&world_buffer, wb.first, wb.second);
	vk->map_buffer(&objects_buffer, objects.data(), sizeof(aligned_vec4) * objects.size());
	vk->map_buffer(&materials_buffer, materials.data(), sizeof(Material) * materials.size());
	vk->map_buffer(&lights_buffer, lights.data(), sizeof(aligned_vec4) * lights.size());
}

// Allocate buffers
void allocate_buffers(Vulkan &vulkan)
{
	// Allocate buffers
	size_t pixel_size = 4 * 800 * 600;
	size_t world_size = world_data_size();
	size_t objects_size = MAX_OBJECT_SIZE * std::max(world.objects.size(), INITIAL_OBJECTS);
	size_t lights_size = MAX_LIGHT_SIZE * std::max(world.lights.size(), INITIAL_LIGHTS);
	size_t materials_size = sizeof(Material) * INITIAL_MATERIALS;

	static const VkBufferUsageFlags buffer_usage =
		VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Create buffers
	vulkan.make_buffer(pixel_buffer,   pixel_size,   buffer_usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	vulkan.make_buffer(world_buffer,   world_size,   buffer_usage);
	vulkan.make_buffer(objects_buffer, objects_size, buffer_usage);
	vulkan.make_buffer(lights_buffer,  lights_size,  buffer_usage);
	vulkan.make_buffer(materials_buffer, materials_size, buffer_usage);
	
	// Add all buffers to deletion queue
	vulkan.push_deletion_task(
		[&](Vulkan *vk) {
			vk->destroy_buffer(pixel_buffer);
			vk->destroy_buffer(world_buffer);
			vk->destroy_buffer(objects_buffer);
			vk->destroy_buffer(lights_buffer);
			vk->destroy_buffer(materials_buffer);
			Logger::ok("[main] Deleted buffers");
		}
	);
}

int main()
{
	// Redirect logger to file
	// Logger::switch_file("mercury.log");

	mercury::Model <VERTEX_TYPE_POSITION> model("resources/teapot.obj");
	model[0] = materials[2];

	world.objects.push_back(std::shared_ptr <mercury::Model <VERTEX_TYPE_POSITION>> (
		new mercury::Model <VERTEX_TYPE_POSITION> (model)
	));

	Logger::ok() << "[main] Loaded model with "
		<< model.mesh_count() << " meshe(s), "
		<< model.mesh(0).vertex_count() << " vertices, "
		<< model.mesh(0).triangle_count() << " triangles" << std::endl;

	// Initialize Vulkan
	Vulkan vulkan;

	// Initialize the buffers
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
	while (!glfwWindowShouldClose(vulkan.window)) {
		glfwPollEvents();
		
		float amplitude = 7.0f;
		glm::vec3 position {
			amplitude * sin(time), 7.0f,
			amplitude * cos(time)
		};

		world.objects[0]->transform.position = position;
		world.lights[0]->transform.position = position;

		// Update buffers
		map_buffers(&vulkan);

		// Show frame
		vulkan.frame();

		// Time
		time += io.DeltaTime;
	}

	vulkan.idle();
}
