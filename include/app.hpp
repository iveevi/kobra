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
private:
	// Vulkan context
	Vulkan *vk;

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
	App(const Info &info) : vk(info.ctx), frame_index(0) {
		// Create surface
		surface = vk->make_surface(info.name,info.width, info.height);

		// Create swapchain
		// TODO: should be passing in window handle into info
		swapchain = vk->make_swapchain(surface);
	}

	// Virtual destructor
	virtual ~App() {}

	// Run application
	void run() {
		while (!glfwWindowShouldClose(vk->window))
			frame(frame_index++);

		vk->idle();
	}

	// Frame function (must be implemented by user)
	virtual void frame(const uint32_t &index) = 0;
};

}

#endif
