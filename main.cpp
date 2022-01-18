#include "vulkan.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector <const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef MERCURY_DEBUG

const bool enableValidationLayers = true;

#else

const bool enableValidationLayers = false;

#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// Generic engine manager
// Functions:
//	- window manager
//	- resource manager
// TODO: will separate into source and header
class Engine {
	// TODO: window should also contain aggregate data
	using Window = GLFWwindow *;

	// Vulkan engine
	Vulkan context;

	std::vector <Window> _windows;

	bool framebufferResized = false;

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		// TODO: this is sus
		auto app = reinterpret_cast<Engine *>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	// Initialize GLFW, and add a window
	void _initalize_glfw() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		Window window = glfwCreateWindow(
			WIDTH, HEIGHT, "Vulkan",
			nullptr, nullptr
		);

		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(
			window, framebufferResizeCallback
		);

		_windows.push_back(window);
	}
public:
	Engine() {
		_initalize_glfw();
		context.initVulkan(_windows[0]);
	}

	~Engine() {
		// Terminate GLFW
		for (Window win : _windows)
			glfwDestroyWindow(win);

		glfwTerminate();
	}

	// Indexing a window
	Window win(size_t i) const {
		return _windows[i];
	}
    
	void mainLoop() {
		while (!glfwWindowShouldClose(win(0))) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(context.device);
	}
    
	void drawFrame() {
		vkWaitForFences(context.device, 1,
			&context.inFlightFences[context.currentFrame],
			VK_TRUE,
			UINT64_MAX
		);

		uint32_t imageIndex;
		// TODO: next line as a Vulkan method
		VkResult result = vkAcquireNextImageKHR(context.device,
				context.swapChain, UINT64_MAX,
				context.imageAvailableSemaphores[context.currentFrame],
				VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			context.recreateSwapChain(win(0));
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		if (context.imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
			vkWaitForFences(context.device, 1,
					&context.imagesInFlight[imageIndex],
					VK_TRUE, UINT64_MAX);
		}

		context.imagesInFlight[imageIndex] = context.inFlightFences[context.currentFrame];

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = {
			context.imageAvailableSemaphores[context.currentFrame]
		};

		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &context.commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = {
			context.renderFinishedSemaphores[context.currentFrame]
		};

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(context.device, 1, &context.inFlightFences[context.currentFrame]);

		if (vkQueueSubmit(context.graphicsQueue, 1, &submitInfo,
				context.inFlightFences[context.currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = {context.swapChain};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(context.presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			context.recreateSwapChain(win(0));
		} else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		context.currentFrame = (context.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}
};

int main()
{
	Engine engine;
	engine.mainLoop();
}
