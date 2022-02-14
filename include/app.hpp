#ifndef APP_H_
#define APP_H_

// Engine headers
#include "backend.hpp"

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

	// Surface
	// TODO: should allow multiple surfaces
	Vulkan::Surface surface;

	// Swapchain
	// TODO: should allow multiple swapchains
	Vulkan::Swapchain swapchain;

	// Frame index
	size_t frame_index;
public:
	// Constructor
	// TODO: constructor for multiple windows?
	App(const Info &info) : ctx(info.ctx), frame_index(0) {
		// Create surface
		surface = ctx->make_surface(info.name,info.width, info.height);

		// Create swapchain
		// TODO: should be passing in window handle into info
		swapchain = ctx->make_swapchain(surface);
	}

	// Virtual destructor
	virtual ~App() {}

	// Run application
	void run() {
		while (!glfwWindowShouldClose(ctx->window)) {
			frame();

			// TODO: mod by max frames in flight
			frame_index = (frame_index + 1) % 2;
		}

		ctx->idle();
	}

	// Frame function (must be implemented by user)
	virtual void frame() = 0;
};

}

#endif
