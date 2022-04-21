// Standard headers
#include <iostream>

// More Vulkan stuff
#include <vulkan/vulkan_core.h>

// Engine headers
#include "../include/backend.hpp"
#include "../include/camera.hpp"

using namespace kobra;

// Cube vertices
float cube_vertices[] = {
	// positions          // texture coords
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
	0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

	-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
};

uint32_t cube_indices[] = {
	0, 1, 2,
	0, 2, 3,
	4, 5, 6,
	4, 6, 7,
	8, 9, 10,
	8, 10, 11,
	12, 13, 14,
	12, 14, 15,
	16, 17, 18,
	16, 18, 19,
	20, 21, 22,
	20, 22, 23,
};

int main()
{
	// Camera
	Camera camera = Camera {
		Transform {
			{0, 0, 5},
			{-0.2, 0, 0}
		},

		Tunings { 45.0f, 800, 800 }
	};

	auto window = Window("Vulkan RT", {200, 200});
	vk::raii::SurfaceKHR surface = make_surface(window);

	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		"VK_KHR_ray_tracing_pipeline"
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	auto phdev = pick_physical_device(predicate);

	std::cout << "Chosen device: " << phdev.getProperties().deviceName << std::endl;
	std::cout << "\tgraphics queue family: " << find_graphics_queue_family(phdev) << std::endl;
	std::cout << "\tpresent queue family: " << find_present_queue_family(phdev, surface) << std::endl;

	auto queue_family = find_queue_families(phdev, surface);
	std::cout << "\tqueue family (G): " << queue_family.graphics << std::endl;
	std::cout << "\tqueue family (P): " << queue_family.present << std::endl;

	auto device = make_device(phdev, queue_family, extensions);

	auto command_pool = vk::raii::CommandPool {
		device,
		{
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			queue_family.graphics
		}
	};

	auto command_buffer = make_command_buffer(device, command_pool);

	auto graphics_queue = vk::raii::Queue { device, queue_family.graphics, 0 };
	auto present_queue = vk::raii::Queue { device, queue_family.present, 0 };

	auto swapchain = Swapchain {phdev, device, surface, window.extent, queue_family};

	auto depth_buffer = DepthBuffer {phdev, device, vk::Format::eD32Sfloat, window.extent};

	KOBRA_LOG_FILE(notify) << "Initialized swapchain and other stuff\n";
}
