// GLFW and Vulkan
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Vulkan for C++
#include <vulkan/vulkan.hpp>

int main()
{
	// Initialize GLFW
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	// Create a window
	GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan RT", nullptr, nullptr);

	// Destroy the window
	glfwDestroyWindow(window);
}
