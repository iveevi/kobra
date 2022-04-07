#ifndef APP_H_
#define APP_H_

// Engine headers
#include "backend.hpp"
#include "coords.hpp"
#include "texture.hpp"
#include "timer.hpp"
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
		io::MouseEventQueue	*mouse_events;
		io::KeyboardEventQueue	*keyboard_events;

		// Immediate IO (keyboard)
		io::Input		*input;

		// Dimensions
		size_t			width;
		size_t			height;

		// Generate screen coords
		coordinates::Screen coordinates(float x, float y) {
			return coordinates::Screen {x, y, width, height};
		}

		// Cursor mode
		void cursor_mode(int mode) {
			glfwSetInputMode(surface.window, GLFW_CURSOR, mode);
		}
	};
protected:
	// Application name
	std::string		name;

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
	App(const Info &info) : name(info.name), frame_index(0) {
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
		}

		context.vk->idle(context.device);

		// Cleanup
		glfwDestroyWindow(surface.window);
	}

	// Frame function (must be implemented by user)
	virtual void frame() = 0;
};

// Base app, includes some more
//	setup upon construction
class BaseApp : public App {
	// Vulkan structures
	VkRenderPass			render_pass;
	VkCommandPool			command_pool;
	std::vector <VkCommandBuffer>	command_buffers;
	VkDescriptorPool		descriptor_pool;

	// Sync objects
	std::vector <VkFence>		in_flight_fences;
	std::vector <VkFence>		images_in_flight;

	std::vector <VkSemaphore>	smph_image_available;
	std::vector <VkSemaphore>	smph_render_finished;

	// Depth resources
	VkImage 			depth_image;
	VkDeviceMemory			depth_image_memory;
	VkImageView			depth_image_view;
public:
	BaseApp(const Info &info, bool depth_testing = false) : App(info) {
		// Create render pass
		// TODO: context method
		render_pass = context.vk->make_render_pass(
			context.phdev,
			context.device,
			swapchain,
			VK_ATTACHMENT_LOAD_OP_CLEAR,
			VK_ATTACHMENT_STORE_OP_STORE,
			depth_testing
		);

		std::vector <VkImageView> extras;
		if (depth_testing) {
			// Create depth image
			VkFormat depth_format = context.find_depth_format();
			context.vk->make_image(context.phdev, context.vk_device(),
				width, height, depth_format,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				depth_image, depth_image_memory
			);

			// Create depth image view
			depth_image_view = context.vk->make_image_view(
				context.vk_device(),
				depth_image, depth_format,
				VK_IMAGE_ASPECT_DEPTH_BIT
			);

			extras.push_back(depth_image_view);
		}

		// Create framebuffers
		context.vk->make_framebuffers(context.device,
			swapchain, render_pass, extras
		);

		// Create command pool
		// TODO: context method
		command_pool = context.vk->make_command_pool(
			context.phdev,
			surface,
			context.device,
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
		);

		// Create descriptor pool
		descriptor_pool = context.vk->make_descriptor_pool(context.device);

		// Copy to window context
		window.command_pool = command_pool;
		window.descriptor_pool = descriptor_pool;

		// Create sync objects
		// TODO: use max frames in flight
		images_in_flight.resize(swapchain.images.size(), VK_NULL_HANDLE);
		for (size_t i = 0; i < 2; i++) {
			in_flight_fences.push_back(context.vk->make_fence(context.device, VK_FENCE_CREATE_SIGNALED_BIT));
			smph_image_available.push_back(context.vk->make_semaphore(context.device));
			smph_render_finished.push_back(context.vk->make_semaphore(context.device));
		}

		// TODO: context method
		context.vk->make_command_buffers(
			context.device,
			command_pool,
			command_buffers,
			swapchain.images.size()
		);
	}

	// Requires a record function
	virtual void record(const VkCommandBuffer &, const VkFramebuffer &) = 0;

	// Possbily override a termination function
	virtual void terminate() {}

	// Present frame
	void present() {
		// Wait for the next image in the swap chain
		vkWaitForFences(
			context.vk_device(), 1,
			&in_flight_fences[frame_index],
			VK_TRUE, UINT64_MAX
		);

		// Acquire the next image from the swap chain
		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(
			context.vk_device(), swapchain.swch, UINT64_MAX,
			smph_image_available[frame_index],
			VK_NULL_HANDLE, &image_index
		);

		// Check if the swap chain is no longer valid
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			// TODO: recreate swap chain
			// _remk_swapchain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			Logger::error("[Vulkan] Failed to acquire swap chain image!");
			throw (-1);
		}

		// Check if the image is being used by the current frame
		if (images_in_flight[image_index] != VK_NULL_HANDLE) {
			vkWaitForFences(
				context.vk_device(), 1,
				&images_in_flight[image_index],
				VK_TRUE, UINT64_MAX
			);
		}

		// Mark the image as in use by this frame
		images_in_flight[image_index] = in_flight_fences[frame_index];

		// Frame submission and synchronization info
		VkSemaphore wait_semaphores[] = {
			smph_image_available[frame_index]
		};

		VkPipelineStageFlags wait_stages[] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
		};

		VkSemaphore signal_semaphores[] = {
			smph_render_finished[frame_index],
		};

		// Record command buffer
		record(command_buffers[image_index], swapchain.framebuffers[image_index]);

		// Create information
		// TODO: method
		VkSubmitInfo submit_info {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = wait_semaphores,
			.pWaitDstStageMask = wait_stages,

			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffers[image_index],

			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signal_semaphores
		};

		// Submit the command buffer
		vkResetFences(context.device.device, 1, &in_flight_fences[frame_index]);
		result = vkQueueSubmit(
			context.device.graphics_queue, 1, &submit_info,
			in_flight_fences[frame_index]
		);

		if (result != VK_SUCCESS) {
			Logger::error("[main] Failed to submit draw command buffer!");
			throw (-1);
		}

		// Present the image to the swap chain
		VkSwapchainKHR swchs[] = {swapchain.swch};

		VkPresentInfoKHR present_info {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signal_semaphores,
			.swapchainCount = 1,
			.pSwapchains = swchs,
			.pImageIndices = &image_index,
			.pResults = nullptr
		};

		result = vkQueuePresentKHR(
			context.device.present_queue,
			&present_info
		);

		/* if (result == VK_ERROR_OUT_OF_DATE_KHR
				|| result == VK_SUBOPTIMAL_KHR
				|| framebuffer_resized) {
			framebuffer_resized = false;
			_remk_swapchain();
		} else*/

		// TODO: check resizing (in app)
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to present swap chain image!");
			throw (-1);
		}
	}

	// Overload frame function
	virtual void frame() override {
		terminate();
		present();
	}
};

}

#endif
