#ifndef BACKEND_H_
#define BACKEND_H_

// TODO: remove the glad diretcory/deps
// GLFW and Vulkan
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
	using Surface = VkSurfaceKHR;
private:
	Instance instance;

	///////////////////////
	// Private functions //
	///////////////////////
	
	// Create Instance
	void _mk_instance() {
		// Create application info
		VkApplicationInfo app_info = {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = "Mercury",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "Mercury",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_0
		};

		// Make instance create info
		VkInstanceCreateInfo create_info = {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &app_info,
			.enabledExtensionCount = 0,
			.ppEnabledExtensionNames = nullptr,
			.enabledLayerCount = 0,
			.ppEnabledLayerNames = nullptr
		};

		// Create instance
		if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create Vulkan instance!");
		}
	}
public:
	// Constructor
	Vulkan() {
		// Create instance
		_mk_instance();
	}
};

#endif
