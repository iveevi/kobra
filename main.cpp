// Standard headers
#include <cstring>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>
// #include <vulkan/vulkan_core.h>

// Local headers
#include "include/mesh.hpp"
#include "profiler.hpp"

using namespace kobra;

int main()
{
	// Redirect logger to file
	// Logger::switch_file("kobra.log");
	
	Mesh <VERTEX_TYPE_NORMAL> mesh;

	// Initialize Vulkan
	Vulkan *vulkan = new Vulkan();

	Profiler *pf = new Profiler();
	ProfilerApplication app {vulkan, pf};
	std::thread thread {
		[&]() { app.run(); }
	};

	thread.join();

	delete vulkan;
}
