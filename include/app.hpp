#ifndef APP_H_
#define APP_H_

// Engine headers
#include "backend.hpp"
#include "timer.hpp"

namespace mercury {

// Application class
class App {
public:
	// Application info structure
	struct Info {
		Vulkan *ctx;

		size_t width;
		size_t height;

		size_t max_frames_in_flight;

		std::string name;
	};
protected:
	// Vulkan context
	Vulkan *ctx;
	
	// Devices
	VkPhysicalDevice		physical_device; // TODO: goes to App
	Vulkan::Device			device;

	// Surface
	// TODO: should allow multiple surfaces
	Vulkan::Surface surface;

	// Swapchain
	// TODO: should allow multiple swapchains
	Vulkan::Swapchain swapchain;

	// Frame information
	Timer		frame_timer;
	double 		frame_time = 0.0;

	size_t		frame_index;
public:
	// Constructor
	// TODO: constructor for multiple windows?
	App(const Info &info) : ctx(info.ctx), frame_index(0) {
		// Create surface
		surface = ctx->make_surface(info.name, info.width, info.height);
		
		// Select the physical device
		physical_device = ctx->select_phdev(surface);

		// Create a logical device
		device = ctx->make_device(physical_device, surface);

		// Create swapchain
		// TODO: should be passing in window handle into info
		swapchain = ctx->make_swapchain(physical_device, device, surface);
	}

	// Virtual destructor
	virtual ~App() {}

	// Run application
	void run() {
		static const double scale = 1e6; 

		// Start timer
		frame_timer.start();
		while (!glfwWindowShouldClose(surface.window)) {
			// Poll events
			glfwPollEvents();

			// Run application frame
			frame();

			// TODO: mod by max frames in flight
			frame_index = (frame_index + 1) % 2;

			// Get frame time
			frame_time = frame_timer.lap()/scale;
		}

		ctx->idle(device);
	}

	// Frame function (must be implemented by user)
	virtual void frame() = 0;
};

}

#endif
