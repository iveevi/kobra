#include "include/app.hpp"
#include "include/backend.hpp"

struct Application {
        vk::raii::Device device = nullptr;
        vk::raii::PhysicalDevice phdev = nullptr;
        vk::raii::SurfaceKHR surface = nullptr;

        kobra::Swapchain swapchain = nullptr;
        kobra::Window *window = nullptr;
};

Application *make_application(const vk::raii::PhysicalDevice &phdev,
                              const vk::Extent2D &extent,
                              const std::string &title)
{
        // Extensions for the application
        static const std::vector <const char *> device_extensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        Application *app = new Application;

        app->phdev = phdev;
        app->window = kobra::make_window(extent, title);
        app->surface = kobra::make_surface(*app->window);

        kobra::QueueFamilyIndices queue_family = kobra::find_queue_families(phdev, app->surface);
        app->device = kobra::make_device(phdev, queue_family, device_extensions);
	app->swapchain = kobra::Swapchain {
                phdev, app->device, app->surface,
                app->window->extent, queue_family
        };

        return app;
}

void delete_application(Application *app)
{
        destroy_window(app->window);
        delete app;
}

// struct dummy : kobra::App {
//         dummy(const vk::raii::PhysicalDevice &phdev_,
//               const std::string &name_,
//               const vk::Extent2D &extent_,
//               const std::vector <const char *> &extensions)
//                 : kobra::App(phdev_, name_, extent_, extensions) {}
//
//         void frame() override {}
// };

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

        Application *app = make_application(phdev, { 800, 600 }, "Ne 2RF 2 Experimental");

        // while (true);

        // Delete application
        delete_application(app);

        return 0;
}
