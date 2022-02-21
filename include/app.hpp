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
	Vulkan::Context context;

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
	App(const Info &info) : frame_index(0) {
		// Create surface
		surface = info.ctx->make_surface(info.name, info.width, info.height);
		
		// Setup the vulkan context
		context.vk = info.ctx;
		context.phdev = context.vk->select_phdev(surface);
		context.device = context.vk->make_device(context.phdev, surface);

		// Create swapchain
		swapchain = context.vk->make_swapchain(
			context.phdev,
			context.device,
			surface
		);
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

		context.vk->idle(context.device);
	}

	// Frame function (must be implemented by user)
	virtual void frame() = 0;
};

}

#endif
