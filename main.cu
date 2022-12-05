#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/common.hpp"
#include "include/ecs.hpp"
#include "include/enums.hpp"
#include "include/io/event.hpp"
#include "include/layers/font_renderer.hpp"
#include "include/layers/gizmo.hpp"
#include "include/layers/objectifier.hpp"
#include "include/layers/optix_tracer.cuh"
#include "include/layers/raster.hpp"
#include "include/layers/raytracer.hpp"
#include "include/logger.hpp"
#include "include/optix/options.cuh"
#include "include/profiler.hpp"
#include "include/project.hpp"
#include "include/renderer.hpp"
#include "include/scene.hpp"
#include "include/transform.hpp"
#include "include/types.hpp"
#include "motion_capture.cuh"
#include "tinyfiledialogs.h"

#include <stb/stb_image_write.h>

using namespace kobra;

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
		return physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (GREEDY)
	auto phdev = pick_physical_device(predicate);

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
