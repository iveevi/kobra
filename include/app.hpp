#ifndef APP_H_
#define APP_H_

// Engine headers
#include "backend.hpp"
#include "timer.hpp"
#include "coords.hpp"
#include "io/event.hpp"
#include "io/input.hpp"

namespace kobra {

// Application class
// (single window)
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

	// Window context
	struct Window {
		Vulkan::Context		context;
		Vulkan::Surface		surface;
		Vulkan::Swapchain	swapchain;

		VkCommandPool		command_pool = VK_NULL_HANDLE;
		VkDescriptorPool	descriptor_pool = VK_NULL_HANDLE;

		// Event based IO
		io::MouseEventQueue *	mouse_events;
		io::KeyboardEventQueue *keyboard_events;

		// Immediate IO (keyboard)
		io::Input *		input;

		// Dimensions
		size_t			width;
		size_t			height;

		// Generate screen coords
		coordinates::Screen coordinates(float x, float y) {
			return coordinates::Screen {x, y, width, height};
		}
	};
protected:
	// This application's window context
	Window			window;

	// Unrolling window context for convenience
	Vulkan::Context		context;
	Vulkan::Surface		surface;
	Vulkan::Swapchain	swapchain;
	io::Input		input;

	// Dimensions
	size_t			width;
	size_t			height;

	// Frame information
	Timer			frame_timer;
	double 			frame_time = 0.0;
	size_t			frame_index;
public:
	// Constructor
	// TODO: constructor for multiple windows?
	App(const Info &info) : frame_index(0) {
		// Create surface
		width = info.width;
		height = info.height;

		Logger::ok() << "width = " << width << ", height = " << height << std::endl;

		surface = info.ctx->make_surface(
			info.name, width, height
		);

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

		// Set GLFW user pointer
		glfwSetWindowUserPointer(
			surface.window, &window
		);

		// Set event callbacks
		glfwSetMouseButtonCallback(surface.window, &io::mouse_button_callback);
		glfwSetCursorPosCallback(surface.window, &io::mouse_position_callback);
		glfwSetKeyCallback(surface.window, &io::keyboard_callback);

		// Setup other window context
		input = io::Input(surface.window);

		// Copy window info
		window.context = context;
		window.surface = surface;
		window.swapchain = swapchain;
		window.mouse_events = new io::MouseEventQueue();
		window.keyboard_events = new io::KeyboardEventQueue();
		window.input = &input;
		window.width = width;
		window.height = height;
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
			// std::cout << "frame_time = " << frame_time << std::endl;
		}

		context.vk->idle(context.device);
	}

	// Frame function (must be implemented by user)
	virtual void frame() = 0;
};

}

#endif
