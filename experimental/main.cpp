// Standard headers
#include <iostream>

// Engine headers
#include "../include/backend.hpp"

using namespace kobra;

int main()
{
	auto window = Window("Vulkan RT", {200, 200});
	vk::raii::SurfaceKHR surface = make_surface(window);

	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		"VK_KHR_surface",
		"VK_KHR_ray_tracing_pipeline"
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	auto phdev = pick_physical_device(predicate);

	std::cout << "Chosen device: " << phdev.getProperties().deviceName << std::endl;
	std::cout << "\tgraphics queue family: " << find_graphics_queue_family(phdev) << std::endl;

	auto queue_family = find_queue_families(phdev, *surface);

	std::cout << "\tqueue family (G): " << queue_family.graphics << std::endl;
	std::cout << "\tqueue family (P): " << queue_family.present << std::endl;
}
