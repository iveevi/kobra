// #include "include/app.hpp"
#include "api.hpp"

// struct dummy : kobra::App {
//         dummy(const vk::raii::PhysicalDevice &phdev_,
//               const std::string &name_,
//               const vk::Extent2D &extent_,
//               const std::vector <const char *> &extensions)
//                 : kobra::App(phdev_, name_, extent_, extensions) {}
//
//         void frame() override {}
// };

struct Application : ApplicationSkeleton {

};

int main()
{
	// Load Vulkan physical device
	auto predicate = [](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev,  {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		});
	};

        // TODO: kobra::pick_first predicate...
	vk::raii::PhysicalDevice phdev = kobra::pick_physical_device(predicate);

        Application *app = new Application;
        make_application(app, phdev, { 800, 600 }, "API Experimental");

        PresentSyncronization sync(app->device, 2);

        uint32_t frame = 0;
        while (true) {
                glfwPollEvents();
                if (glfwWindowShouldClose(app->window->handle))
                        break;

                SurfaceOperation op;
                op = acquire_image(app->device, app->swapchain.swapchain, sync, frame);
                op = present_image(app->present_queue, app->swapchain.swapchain, sync, op.index);
        }

        // Delete applica
        destroy_application(app);
        delete app;

        return 0;
}
