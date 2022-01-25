#ifndef MERCURY_VULKAN_H_
#define MERCURY_VULKAN_H_

// Standard headers
#include <optional>
#include <set>
#include <string>
#include <vector>

// GLFW and Vulkan
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

extern const bool enableValidationLayers;

const int MAX_FRAMES_IN_FLIGHT = 2;

extern const std::vector <const char *> validationLayers;
extern const std::vector <const char *> deviceExtensions;

extern VkResult CreateDebugUtilsMessengerEXT(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT*, const VkAllocationCallbacks*, VkDebugUtilsMessengerEXT*);
extern void DestroyDebugUtilsMessengerEXT(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks *);

struct QueueFamilyIndices {
    std::optional <uint32_t> graphicsFamily;
    std::optional <uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// All vulkan related procedures
struct Vulkan {
	// TODO: what needs to be private?
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphicsQueue;
	VkQueue presentQueue;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;

	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	VkCommandPool commandPool;
	std::vector <VkCommandBuffer> commandBuffers;

	std::vector <VkSemaphore> imageAvailableSemaphores;
	std::vector <VkSemaphore> renderFinishedSemaphores;
	std::vector <VkFence> inFlightFences;
	std::vector <VkFence> imagesInFlight;

	// TODO: these two should be in the engine class, not this one
	size_t currentFrame = 0;
	bool framebufferResized = false;
        
	void initVulkan(GLFWwindow *);
        void cleanupSwapChain();
        void cleanup();
        void recreateSwapChain(GLFWwindow *);
        void createInstance();
        void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
        void setupDebugMessenger();
        void createSurface(GLFWwindow *);
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createSwapChain(GLFWwindow *);
        void createImageViews();
        void createRenderPass();
        void createGraphicsPipeline();
        void createFramebuffers();
        void createCommandPool();
        void createCommandBuffers();
        void createSyncObjects();
        VkShaderModule createShaderModule(const std::vector<char> &code);
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);
        VkExtent2D chooseSwapExtent(GLFWwindow *, const VkSurfaceCapabilitiesKHR &capabilities);
        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
        bool isDeviceSuitable(VkPhysicalDevice device);
        bool checkDeviceExtensionSupport(VkPhysicalDevice device);
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
        std::vector<const char *> getRequiredExtensions();
        bool checkValidationLayerSupport();

        static std::vector <char> read(const std::string &filename);
	
	static VKAPI_ATTR VkBool32 VKAPI_CALL
	debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT
		messageSeverity, VkDebugUtilsMessageTypeFlagsEXT
		messageType, const
		VkDebugUtilsMessengerCallbackDataEXT
		*pCallbackData, void *pUserData);
};

#endif
