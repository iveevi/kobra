#ifndef APP_H_
#define APP_H_

// Engine headers
#include "backend.hpp"
#include "coords.hpp"
// #include "texture.hpp"
#include "timer.hpp"
#include "io/event.hpp"
#include "io/input.hpp"

namespace kobra {

// Application class
// (single window)
class App {
public:
	struct IO {
		io::MouseEventQueue	mouse_events;
		io::KeyboardEventQueue	keyboard_events;
		io::Input		input;
	};
protected:
	// Stored window info
	vk::raii::PhysicalDevice	phdev;
	vk::raii::Device 		device;
	vk::raii::SurfaceKHR		surface = nullptr;

	Swapchain 			swapchain = nullptr;
	Window				window;

	// IO info
	IO				io;

	// Application name
	std::string			name;

	// Dimensions
	vk::Extent2D			extent;

	// Frame information
	Timer			frame_timer;
	double 			frame_time = 0.0;
	size_t			frame_index;

	// Termination status
	bool			terminated = false;

	// Generate screen coords
	coordinates::Screen coordinates(float x, float y) {
		return coordinates::Screen {x, y, extent.width, extent.height};
	}

public:
	// Constructor
	App(const vk::raii::PhysicalDevice &phdev, vk::raii::Device &device,
			const vk::Extent2D &extent_)
			: phdev(phdev),
			device {std::move(device)},
			extent(extent_) {
		// TODO: move to source file

		// First create the window
		window = Window("Kobra Application", extent);

		// Create Vulkan surface and swapchain
		surface = make_surface(window);
		auto queue_family = find_queue_families(phdev, surface);
		swapchain = Swapchain {phdev, device, surface, window.extent, queue_family};

		// Set user pointer
		glfwSetWindowUserPointer(window.handle, &io);

		// Set event callbacks
		glfwSetMouseButtonCallback(window.handle, &io::mouse_button_callback);
		glfwSetCursorPosCallback(window.handle, &io::mouse_position_callback);
		glfwSetKeyCallback(window.handle, &io::keyboard_callback);
	}

	// Virtual destructor
	virtual ~App() {}

	// Run application
	void run() {
		static const double scale = 1e6;

		// Start timer
		frame_timer.start();
		while (!glfwWindowShouldClose(window.handle)) {
			// Check if manually terminated
			if (terminated)
				break;

			// Poll events
			glfwPollEvents();

			// Run application frame
			frame();

			// TODO: mod by max frames in flight
			frame_index = (frame_index + 1) % 2;

			// Get frame time
			frame_time = frame_timer.lap()/scale;
		}

		// Idle till all frames are finished
		device.waitIdle();

		// Cleanup
		glfwDestroyWindow(window.handle);
	}

	// Manually terminate application
	void terminate_now() {
		terminated = true;
	}

	// Frame function (must be implemented by user)
	virtual void frame() = 0;
};

// Base app, includes some more setup upon construction
class BaseApp : public App {
protected:
	// Vulkan structures
	vk::raii::RenderPass			render_pass = nullptr;
	std::vector <vk::raii::Framebuffer>	framebuffers;

	vk::raii::CommandPool			command_pool = nullptr;
	vk::raii::DescriptorPool		descriptor_pool = nullptr;

	vk::raii::Queue				graphics_queue = nullptr;
	vk::raii::Queue				present_queue = nullptr;

	std::array <vk::raii::CommandBuffer, 2>	command_buffers = {nullptr, nullptr};

	DepthBuffer				depth_buffer = nullptr;

	// Syncro objects
	struct FrameData {
		vk::raii::Semaphore		present_completed = nullptr;
		vk::raii::Semaphore		render_completed = nullptr;
		vk::raii::Fence			fence = nullptr;

		// Default constructor
		FrameData() = default;

		// Construct from device
		FrameData(vk::raii::Device &device) {
			present_completed = vk::raii::Semaphore(device, {});
			render_completed = vk::raii::Semaphore(device, {});
			fence = vk::raii::Fence(device, {vk::FenceCreateFlagBits::eSignaled});
		}
	};

	std::vector <FrameData>			frames;
public:
	// TODO: is attachment load op needed?
	BaseApp(const vk::raii::PhysicalDevice &phdev_,
			vk::raii::Device &device_,
			const vk::Extent2D &extent_,
			const vk::AttachmentLoadOp &load = vk::AttachmentLoadOp::eClear)
			: App(phdev_, device_, extent_) {
		// Create the depth buffer
		depth_buffer = DepthBuffer {
			phdev, device,
			vk::Format::eD32Sfloat,
			extent
		};

		// Create render pass
		render_pass = make_render_pass(device,
			swapchain.format,
			depth_buffer.format,
			load
		);

		// Create the framebuffers
		framebuffers = make_framebuffers(device,
			render_pass,
			swapchain.image_views,
			&depth_buffer.view,
			extent
		);

		// Create command pool
		command_pool = vk::raii::CommandPool {
			device, {
				vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				find_graphics_queue_family(phdev)
			}
		};

		// Create command buffers
		// TODO: function for muliple
		command_buffers = std::array <vk::raii::CommandBuffer, 2> {
			make_command_buffer(device, command_pool),
			make_command_buffer(device, command_pool)
		};

		// Create descriptor pool
		descriptor_pool = make_descriptor_pool(
			device, {
				// TODO: static const
				{vk::DescriptorType::eSampler, 1024},
				{vk::DescriptorType::eCombinedImageSampler, 1024},
				{vk::DescriptorType::eSampledImage, 1024},
				{vk::DescriptorType::eStorageImage, 1024},
				{vk::DescriptorType::eUniformTexelBuffer, 1024},
				{vk::DescriptorType::eStorageTexelBuffer, 1024},
				{vk::DescriptorType::eUniformBuffer, 1024},
				{vk::DescriptorType::eStorageBuffer, 1024},
				{vk::DescriptorType::eUniformBufferDynamic, 1024},
				{vk::DescriptorType::eStorageBufferDynamic, 1024},
				{vk::DescriptorType::eInputAttachment, 1024}
			}
		);

		// Create syncro objects
		frames = std::vector <FrameData> (framebuffers.size());
		for (auto &frame : frames)
			frame = FrameData {device};

		// Get queues
		auto queue_family = find_queue_families(phdev, surface);

		graphics_queue = vk::raii::Queue {device, queue_family.graphics, 0};
		present_queue = vk::raii::Queue {device, queue_family.present, 0};
	}

	// Requires a record function
	virtual void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer &) = 0;

	// Possbily override a termination function
	virtual void terminate() {}

	// Present frame
	// TODO: add a present method in backend
	void present() {
		// Stage flags
		static const vk::PipelineStageFlags stage_flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		// Result from Vulkan functions
		vk::Result result;

		// Image index
		uint32_t image_index;

		// Get the current command buffer
		const auto &command_buffer = command_buffers[frame_index];

		// Acquire the next image from the swapchain
		std::tie(result, image_index) = swapchain.swapchain.acquireNextImage(
			std::numeric_limits <uint64_t>::max(),
			*frames[frame_index].present_completed
		);

		KOBRA_ASSERT(result == vk::Result::eSuccess, "Failed to acquire next image");

		// Wait for the previous frame to finish rendering
		while (vk::Result(device.waitForFences(
			*frames[frame_index].fence,
			true,
			std::numeric_limits <uint64_t>::max()
		)) == vk::Result::eTimeout);

		// Then reset the fence
		device.resetFences(*frames[frame_index].fence);

		// Record the command buffer
		record(command_buffer, framebuffers[image_index]);

		// Submit the command buffer
		vk::SubmitInfo submit_info {
			1, &*frames[frame_index].present_completed,
			&stage_flags,
			1, &*command_buffer,
			1, &*frames[frame_index].render_completed
		};

		graphics_queue.submit(submit_info, *frames[frame_index].fence);

		// Present the image
		vk::PresentInfoKHR present_info {
			*frames[frame_index].render_completed,
			*swapchain.swapchain,
			image_index
		};

		result = present_queue.presentKHR(present_info);

		KOBRA_ASSERT(result == vk::Result::eSuccess, "Failed to present image");
	}

	// Overload frame function
	virtual void frame() override {
		terminate();
		present();
	}
};

}

#endif
