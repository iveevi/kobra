#include "include/project.hpp"
#include "motion_capture.cuh"

int main()
{
	// Load Vulkan physical device
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (GREEDY)
	auto phdev = kobra::pick_physical_device(predicate);

	std::cout << "Extensions:" << std::endl;
	for (auto str : extensions)
		std::cout << "\t" << str << std::endl;

	// Load the project
	kobra::Project project = kobra::Project::load(".kobra/project");

	// Create and launch the application
	MotionCapture app(phdev, {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	}, project.scene);

	app.run();
}
