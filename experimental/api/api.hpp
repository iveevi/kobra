#pragma once

// Engine headers
#include "include/backend.hpp"

struct PresentSyncronization {
        std::vector <vk::raii::Semaphore> image_available;
        std::vector <vk::raii::Semaphore> render_finished;
        std::vector <vk::raii::Fence> in_flight;

        PresentSyncronization(const vk::raii::Device &device, uint32_t frames_in_flight) {
                // Default semaphores
                vk::SemaphoreCreateInfo semaphore_info;

                // Signaled fences
                vk::FenceCreateInfo fence_info;
                fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

                for (uint32_t i = 0; i < frames_in_flight; i++) {
                        image_available.push_back(device.createSemaphore(semaphore_info));
                        render_finished.push_back(device.createSemaphore(semaphore_info));
                        in_flight.push_back(device.createFence(fence_info));
                }
        }
};

struct SurfaceOperation {
        enum {
                eOk,
                eResize,
                eFailed
        } status;

        uint32_t index;
};

SurfaceOperation acquire_image(const vk::raii::Device &device,
                const vk::raii::SwapchainKHR &swapchain,
                const PresentSyncronization &sync,
                uint32_t frame)
{
        // Wait for previous frame to finish
        device.waitForFences(*sync.in_flight[frame], VK_TRUE, UINT64_MAX);

        // Acquire image
        auto [result, image_index] = swapchain.acquireNextImage(UINT64_MAX, *sync.image_available[frame], nullptr);
        if (result == vk::Result::eErrorOutOfDateKHR) {
                std::cerr << "Swapchain out of date" << std::endl;
                return { SurfaceOperation::eResize, 0 };
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
                std::cerr << "Failed to acquire swapchain image" << std::endl;
                return { SurfaceOperation::eFailed, 0 };
        }

        // Reset fence to prepare for next frame
        device.resetFences(*sync.in_flight[frame]);

        return { SurfaceOperation::eOk, image_index };
}

SurfaceOperation present_image(const vk::raii::Queue &queue,
                const vk::raii::SwapchainKHR &swapchain,
                const PresentSyncronization &sync,
                uint32_t index)
{
        vk::PresentInfoKHR present_info {
                *sync.render_finished[index],
                *swapchain,
                index
        };

        try {
                queue.presentKHR(present_info);
        } catch (vk::OutOfDateKHRError &e) {
                std::cerr << "Swapchain out of date" << std::endl;
                return { SurfaceOperation::eResize, 0 };
        }

        return { SurfaceOperation::eOk, 0 };
}

struct ApplicationSkeleton {
        vk::raii::Device device = nullptr;
        vk::raii::PhysicalDevice phdev = nullptr;
        vk::raii::SurfaceKHR surface = nullptr;

        vk::raii::Queue graphics_queue = nullptr;
        vk::raii::Queue present_queue = nullptr;

        kobra::Swapchain swapchain = nullptr;
        kobra::Window *window = nullptr;
};

void make_application(ApplicationSkeleton *app,
                const vk::raii::PhysicalDevice &phdev,
                const vk::Extent2D &extent,
                const std::string &title)
{
        // Extensions for the application
        static const std::vector <const char *> device_extensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        app->phdev = phdev;
        app->window = kobra::make_window(extent, title);
        app->surface = kobra::make_surface(*app->window);

        kobra::QueueFamilyIndices queue_family = kobra::find_queue_families(phdev, app->surface);
        app->device = kobra::make_device(phdev, queue_family, device_extensions);
	app->swapchain = kobra::Swapchain {
                phdev, app->device, app->surface,
                app->window->extent, queue_family
        };

        kobra::QueueFamilyIndices indices = kobra::find_queue_families(phdev, app->surface);
        app->graphics_queue = vk::raii::Queue { app->device, queue_family.graphics, 0 };
        app->present_queue = vk::raii::Queue { app->device, queue_family.present, 0 };
}

void destroy_application(ApplicationSkeleton *app)
{
        destroy_window(app->window);
}
