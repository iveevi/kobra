// Standard headers
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "backend.hpp"

int main()
{
	// Create GLFW context
	GLFW glfw("Vulkan");

	// Create Vulkan context
	Vulkan vk;

	// Main loop
	while (!glfwWindowShouldClose(glfw[0])) {
		// Poll events
		glfwPollEvents();
	}

	return 0;
}
