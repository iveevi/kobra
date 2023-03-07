#include "../include/app.hpp"

namespace kobra {

/////////
// App //
/////////


// Constructor
App::App(const vk::raii::PhysicalDevice &phdev_,
		const std::string &name_,
		const vk::Extent2D &extent_,
		const std::vector <const char *> &extensions)
		: phdev(phdev_),
		window(name_, extent_),
		frame_index(0)
{
	surface = make_surface(window);
	auto queue_family = find_queue_families(phdev, surface);

	// TODO: extensions as a device
	std::cout << "Window extent: " << window.m_extent.width << "x" << window.m_extent.height << std::endl;
	device = make_device(phdev, queue_family, extensions);
	swapchain = Swapchain {phdev, device, surface, window.m_extent, queue_family};

	KOBRA_LOG_FUNC(Log::OK) << "Swapchain format: " << vk::to_string(swapchain.format) << std::endl;

	// GLFW things
	glfwSetWindowUserPointer(window.m_handle, &io);
	glfwSetMouseButtonCallback(window.m_handle, &io::mouse_button_callback);
	glfwSetCursorPosCallback(window.m_handle, &io::mouse_position_callback);
	glfwSetKeyCallback(window.m_handle, &io::keyboard_callback);

	// TODO: store memory requirements

	// Initialize IO info
	io.input = new io::Input(window.m_handle);
}

// Virtual destructor
App::~App()
{
	delete io.input;
}

// Run application
void App::run()
{
	static const double scale = 1e6;

	// Start timer
	frame_timer.start();
	while (!glfwWindowShouldClose(window.m_handle)) {
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

	KOBRA_LOG_FILE(Log::OK) << "App successfully terminated.\n";

	// Idle till all frames are finished
	device.waitIdle();
}

// Manually terminate application
void App::terminate_now()
{
	terminated = true;
}

// Protected methods
coordinates::Screen App::coordinates(float x, float y)
{
	return coordinates::Screen {
		x, y,
		window.m_extent.width, window.m_extent.height
	};
}

Device App::get_device()
{
	return Device {
		.phdev = &phdev,
		.device = &device
	};
}

//////////////
// Base App //
//////////////

// Constructor
// TODO: is attachment load op needed?
BaseApp::BaseApp(const vk::raii::PhysicalDevice &phdev_,
		const std::string &name_,
		const vk::Extent2D &extent_,
		const std::vector <const char *> &extensions,
		const vk::AttachmentLoadOp &load)
		: App(phdev_, name_, extent_, extensions)
{
	// Create the depth buffer
	std::cout << "Creating depth buffer: " << window.m_extent.width << "x" << window.m_extent.height << std::endl;
	depth_buffer = std::move(DepthBuffer {
		phdev, device,
		vk::Format::eD32Sfloat,
		window.m_extent
	});

	// Initialize the texture loader
	m_texture_loader = std::move(TextureLoader {get_device()});

	// Create render pass
	// TODO: we dont need to create a base render pass (whos gonna
	// use it?)
	render_pass = make_render_pass(device,
		{swapchain.format}, {load},
		depth_buffer.format, load
	);

	// Create the framebuffers
	framebuffers = make_framebuffers(device,
		render_pass,
		swapchain.image_views,
		&depth_buffer.view,
		window.m_extent
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
	auto frames_ = std::vector <FrameData> (framebuffers.size());
	for (auto &frame : frames_)
		frame = FrameData {device};

	frames = std::move(frames_);

	// Get queues
	auto queue_family = find_queue_families(phdev, surface);

	graphics_queue = vk::raii::Queue {device, queue_family.graphics, 0};
	present_queue = vk::raii::Queue {device, queue_family.present, 0};
}

// Possbily override an after-present function
void BaseApp::after_present() {}

// Possbily override a termination function
void BaseApp::terminate() {}

// Possbily override a resize function
void BaseApp::resize(const vk::Extent2D &) {}

// Present frame
// TODO: add a present method in backend
void BaseApp::present()
{
	// Stage flags
	static const vk::PipelineStageFlags stage_flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	// Perform sync tasks if needed
	if (sync_queue.size() > 0) {
		graphics_queue.waitIdle();

		while (sync_queue.size() > 0)
			sync_queue.do_pop(false);
	}

	// Result from Vulkan functions
	vk::Result result;

	// Image index
	uint32_t image_index;

	// Get the current command buffer
	const auto &command_buffer = command_buffers[frame_index];

	// Wait for the previous frame to finish rendering
	while (vk::Result(device.waitForFences(
		*frames[frame_index].fence,
		true,
		std::numeric_limits <uint64_t>::max()
	)) == vk::Result::eTimeout);

	// Acquire the next image from the swapchain
	std::tie(result, image_index) = swapchain.swapchain.acquireNextImage(
		std::numeric_limits <uint64_t>::max(),
		*frames[frame_index].present_completed
	);

	// KOBRA_ASSERT(result == vk::Result::eSuccess, "Failed to acquire next image");
	if (result == vk::Result::eErrorOutOfDateKHR) {
		// TODO: need to also resize if the callback to glfw ran:
		// some drivers wont return error out of date
		recreate_swapchain();
		return;
	} else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
		throw std::runtime_error("Failed to acquire next image");
	}

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

	try {
		result = present_queue.presentKHR(present_info);
	} catch (const vk::OutOfDateKHRError &e) {
		recreate_swapchain();
	}

	// Afer present actions
	after_present();
}

// Overload frame function
void BaseApp::frame()
{
	terminate();
	present();
}

///////////////////////
// Protected methods //
///////////////////////

Context BaseApp::get_context()
{
	return Context {
		.phdev = &phdev,
		.device = &device,
		.command_pool = &command_pool,
		.descriptor_pool = &descriptor_pool,
		.sync_queue = &sync_queue,
		.extent = window.m_extent,
		.swapchain_format = swapchain.format,
		.depth_format = depth_buffer.format,
		.texture_loader = &m_texture_loader
	};
}

void BaseApp::recreate_swapchain()
{
	// Update the current extent
	int width = 0;
	int height = 0;

	glfwGetFramebufferSize(window.m_handle, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window.m_handle, &width, &height);
		glfwWaitEvents();
	}

	window.m_extent.width = width;
	window.m_extent.height = height;

	// Wait for the device to be idle
	device.waitIdle();

	// Recreate the swapchain
	auto queue_family = find_queue_families(phdev, surface);
	swapchain = Swapchain {
		phdev, device, surface, window.m_extent,
		queue_family, &swapchain.swapchain
	};

	// Transition all images to the correct layout
	submit_now(device, graphics_queue, command_pool,
		[&](const vk::raii::CommandBuffer &cmd) {
			std::vector <vk::ImageMemoryBarrier> barriers;
			for (const vk::Image &image : swapchain.images) {
				barriers.emplace_back(vk::ImageMemoryBarrier {
					{},
					vk::AccessFlagBits::eColorAttachmentWrite,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::ePresentSrcKHR,
					VK_QUEUE_FAMILY_IGNORED,
					VK_QUEUE_FAMILY_IGNORED,
					image,
					vk::ImageSubresourceRange {
						vk::ImageAspectFlagBits::eColor,
						0, 1, 0, 1
					}
				});
			}

			cmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::PipelineStageFlagBits::eColorAttachmentOutput,
				{}, {}, {}, barriers
			);
		}
	);

	// Recreate the frame and depth buffers
	depth_buffer = std::move(DepthBuffer {
		phdev, device,
		vk::Format::eD32Sfloat,
		window.m_extent
	});

	// TODO: remove the framebuffers from here...
	// TODO: remove the base app abstraction, and
	// create a present method in the backend
	framebuffers = make_framebuffers(device,
		render_pass,
		swapchain.image_views,
		&depth_buffer.view,
		window.m_extent
	);

	// Call possibly overriden resize function
	resize(window.m_extent);
}

}
