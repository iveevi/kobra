#ifndef BACKEND_H_
#define BACKEND_H_

// Standard headers
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// TODO: remove the glad diretcory/deps
// GLFW and Vulkan
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Engine headers
#include "include/logger.hpp"

// GLFW context type
class GLFW {
public:
	// Public aliases
	using Window = GLFWwindow *;
private:
	Window win = nullptr;

	// Array of all windows
	std::vector <Window> windows;

	///////////////////////
	// Private functions //
	///////////////////////
	
	// Create a window
	Window _mk_win(const std::string &title, int width, int height) {
		// Create window
		Window win = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		// Check if window was created
		if (win == nullptr) {
			throw std::runtime_error("Failed to create GLFW window");
		}
		// Add window to array
		windows.push_back(win);

		// Return window
		return win;
	}

	// Set current window
	void _set_current(Window win) {
		glfwMakeContextCurrent(win);
		this->win = win;
	}
public:
	// Constructor
	GLFW(const std::string &title, int width = 800, int height = 600) {
		// Initialize GLFW
		if (!glfwInit()) {
			throw std::runtime_error("Failed to initialize GLFW!");
		}

		// Hints
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		// Set GLFW error callback
		// TODO: callback to logger
		glfwSetErrorCallback([](int error, const char *description) {
			std::cerr << "GLFW Error: " << description << std::endl;
		});

		// Create window
		_mk_win(title, width, height);

		// Set current window
		_set_current(win);
	}

	// Destructor
	~GLFW() {
		// Destroy all windows
		for (auto w : windows)
			glfwDestroyWindow(w);

		// Terminate GLFW
		glfwTerminate();
	}

	// Indexing operator
	Window operator[](size_t index) {
		return windows[index];
	}
};

// Vulkan context type
class Vulkan {
public:
	// Public aliases
	using Instance = VkInstance;
	using Device = VkDevice;
	using PhysicalDevice = VkPhysicalDevice;
	using Surface = VkSurfaceKHR;

	using Args = std::vector <const char *>;
private:
	Instance instance;

        // Validation layers
        static const Args validation_layers;

#ifdef MERCURY_VALIDATION_LAYERS

	// Enabling validation layers
	static constexpr bool enable_validation_layers = true;

#else

	// Disabling validation layers
	static constexpr bool enable_validation_layers = false;

#endif

	///////////////////////
	// Private functions //
	///////////////////////
	
	// Create Instance
	void _mk_instance() {
		// Check if validation layers are enabled
		// and if so, add confirm
		if (enable_validation_layers && !_check_validation_layers()) {
			Logger::error("[Vulkan] Validation layers requested, but not available!");
			throw int(-1);
		}

		// Create application info
		VkApplicationInfo app_info = {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = "Mercury",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "Mercury",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_0
		};

                // Get GLFW extensions
                uint32_t glfw_ext_count = 0;
                const char **glfw_ext = nullptr;

                glfw_ext = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

		// Make instance create info
		VkInstanceCreateInfo create_info = {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &app_info
		};

		// Add extensions
		Args extensions = _get_required_extensions();
		create_info.enabledExtensionCount = static_cast
			<uint32_t> (extensions.size());
		create_info.ppEnabledExtensionNames = extensions.data();

		// Add validation layers if enabled
		if (enable_validation_layers) {
			create_info.enabledLayerCount = static_cast
				<uint32_t> (validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		}

		// Create instance
		if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create Vulkan instance!");
			throw int(-1);
		}
	}

	///////////////////////////////
	// Device and queue handling //
	///////////////////////////////
	
	// Chosen physical device
	PhysicalDevice physical_device = VK_NULL_HANDLE;

	// Queue family info
	//	contains indices to queue families
	struct QueueFamilyInfo {
		std::optional <uint32_t> graphics;

		// As a boolean status
		operator bool() const {
			return graphics.has_value();
		}
	};

	// Find queue family info
	QueueFamilyInfo _find_queue_family_info(PhysicalDevice phdev) {
		// Find queue families
		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(
			phdev,
			&queue_family_count,
			nullptr
		);

		// Allocate memory for queue families
		std::vector <VkQueueFamilyProperties> queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(
			physical_device,
			&queue_family_count,
			queue_families.data()
		);

		// Find graphics queue family
		QueueFamilyInfo info = {
			.graphics = -1
		};

		for (uint32_t i = 0; i < queue_family_count; i++) {
			if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				info.graphics = i;
				break;
			}
		}

		// Return queue family info
		return info;
	}
	
	// Check if device is favorable
	bool _is_device_favorable(const PhysicalDevice &phdev) {
		// Get queue family info
		return _find_queue_family_info(phdev);
	}

	// Logical devices, etc.
	Device logical_device;

	// Create logical device
	void _mk_logical_device(PhysicalDevice phdev) {
		// Find queue family info
		QueueFamilyInfo info = _find_queue_family_info(phdev);

		// Make queue create info
		float qpriority = 1.0f;
		VkDeviceQueueCreateInfo queue_create_info = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = info.graphics.value(),
			.queueCount = 1,
			.pQueuePriorities = &qpriority
		};
	}

	// Choose the device
	// NOTE: first one for now
	void _choose_physical_device() {
		// Get physical devices
		uint32_t phdev_count = 0;
		vkEnumeratePhysicalDevices(instance, &phdev_count, nullptr);

		// Check if no devices
		if (phdev_count == 0) {
			Logger::error("[Vulkan] No physical devices found!");
			throw int(-1);
		}

		// Create array of physical devices
		std::vector <PhysicalDevice> phdevs(phdev_count);
		vkEnumeratePhysicalDevices(instance, &phdev_count, phdevs.data());

		// Iterate through devices
		for (auto pd : phdevs) {
			// Check if device is favorable
			if (_is_device_favorable(pd)) {
				// Set device
				physical_device = pd;
				return;
			}
		}

		// If no favorable device found
		if (physical_device == VK_NULL_HANDLE) {
			Logger::error("[Vulkan] No favorable device found!");
			throw int(-1);
		}
	}

	/////////////////////////////////////
	// Debugging and validation layers //
	/////////////////////////////////////
	
	// Debug messenger
	VkDebugUtilsMessengerEXT debug_messenger;

	// Helper function for messenger creation
	static VkResult _mk_debug_utils_messenger_ext(
			VkInstance instance,
			const VkDebugUtilsMessengerCreateInfoEXT *create_info,
			const VkAllocationCallbacks *allocator,
			VkDebugUtilsMessengerEXT *debug_messenger) {
		auto f = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
				instance, "vkCreateDebugUtilsMessengerEXT"
		);

		if (f != nullptr)
			return f(instance, create_info, allocator, debug_messenger);

		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	// Destroying debug messenger
	static void _free_debug_messenger(VkInstance instance,
			VkDebugUtilsMessengerEXT debug_messenger,
			const VkAllocationCallbacks *allocator) {
		auto f = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
				instance, "vkDestroyDebugUtilsMessengerEXT"
		);

		if (f != nullptr)
			f(instance, debug_messenger, allocator);
	}

	// Create debug messenger
	void _mk_debug_messenger() {
		// Quit if validation layers are not enabled
		if (!enable_validation_layers)
			return;

		// Create debug messenger create info
		VkDebugUtilsMessengerCreateInfoEXT create_info = {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
			.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
			.pfnUserCallback = _debug_callback,
			.pUserData = nullptr
		};

		// Create debug messenger
		VkResult result = _mk_debug_utils_messenger_ext(
			instance, &create_info,
			nullptr, &debug_messenger
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create debug messenger!");
			throw int(-1);
		}
	}

	// Get required extensions
	Args _get_required_extensions() {
		// Get GLFW extensions
		uint32_t glfw_ext_count = 0;
		const char **glfw_ext = nullptr;

		glfw_ext = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

		// Make extensions array
		Args extensions(glfw_ext, glfw_ext + glfw_ext_count);

		// Add validation layers if enabled
		if (enable_validation_layers)
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

		// Return extensions
		return extensions;
	}

	// Check validation layers
	bool _check_validation_layers() {
		// Get available layers
		uint32_t layer_count;
		vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

		// Allocate memory for layers
		std::vector <VkLayerProperties> available_layers(layer_count);

		// Get available layers
		vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

		// Check if all layers are available
		for (const char *layer_name : validation_layers) {
			bool layer_found = false;

			// Check if layer is available
			for (const auto &layer_props : available_layers) {
				if (std::strcmp(layer_name, layer_props.layerName) == 0) {
					layer_found = true;
					break;
				}
			}

			// Check if layer was found
			if (!layer_found) {
				return false;
			}
		}

		// All layers are available
		return true;
	}

	// Message callback
	static VKAPI_ATTR VkBool32 VKAPI_CALL _debug_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT severity,
		VkDebugUtilsMessageTypeFlagsEXT type,
		const VkDebugUtilsMessengerCallbackDataEXT *data,
		void *user_data) {
		// Switch on severity
		switch (severity) {
		case VK_DEBUG_REPORT_ERROR_BIT_EXT:
			Logger::error(data->pMessage);
			break;
		case VK_DEBUG_REPORT_WARNING_BIT_EXT:
		case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
			Logger::warn(data->pMessage);
			break;
		case VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
		case VK_DEBUG_REPORT_DEBUG_BIT_EXT:
		default:
			Logger::notify(data->pMessage);
			break;
		}

		return VK_FALSE;
	}
public:
	// Constructor
	Vulkan() {
		// Create instance
		_mk_instance();

		// Create debugger instance
		_mk_debug_messenger();

		// Choose physical device
		_choose_physical_device();

		// Create logical device
		_mk_logical_device(physical_device);
	}

	// Destructor
	~Vulkan() {
		// Destroy debug messenger
		if (debug_messenger != nullptr)
			_free_debug_messenger(instance, debug_messenger, nullptr);

		// Destroy instance
		vkDestroyInstance(instance, nullptr);
	}
};

// Static variables
// TODO: source file
const Vulkan::Args Vulkan::validation_layers = {
	"VK_LAYER_KHRONOS_validation"
};

#endif
