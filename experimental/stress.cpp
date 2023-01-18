#include "../include/app.hpp"
#include "../include/backend.hpp"

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
	
	vk::raii::PhysicalDevice phdev = kobra::pick_physical_device(predicate);

	struct StressApp : public kobra::BaseApp {
		StressApp(const vk::raii::PhysicalDevice &phdev,
				const std::vector <const char *> extensions)
				: kobra::BaseApp {
					phdev, "Stress Test",
					vk::Extent2D {500, 500},
					extensions
				} {}

		void record(const vk::raii::CommandBuffer &,
				const vk::raii::Framebuffer &) override {}
	};

	StressApp app {
		phdev,
		{VK_KHR_SWAPCHAIN_EXTENSION_NAME},
	};
}
