#ifndef KOBRA_APP_H_
#define KOBRA_APP_H_

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
		io::Input		*input;
	};
protected:
	// Stored window info
	vk::raii::PhysicalDevice	phdev = nullptr;
	vk::raii::Device 		device = nullptr;
	vk::raii::SurfaceKHR		surface = nullptr;

	Swapchain 			swapchain = nullptr;
	Window				window;

	// IO info
	IO				io;

	// Application name
	std::string			name;

	// Frame information
	Timer				frame_timer;
	double				frame_time = 0.0;
	size_t				frame_index;

	// Termination status
	bool				terminated = false;

	// Generate screen coords
	coordinates::Screen coordinates(float x, float y);

	// Generating aux structures
	Device get_device();
public:
	// Constructor
	App(const vk::raii::PhysicalDevice &,
			const std::string &,
			const vk::Extent2D &,
			const std::vector <const char *> &);

	// Virtual destructor
	virtual ~App();

	// Run application
	void run();

	// Manually terminate application
	void terminate_now();

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

	TextureLoader				m_texture_loader;

	// Syncro objects
	struct FrameData {
		vk::raii::Semaphore		present_completed = nullptr;
		vk::raii::Semaphore		render_completed = nullptr;
		vk::raii::Fence			fence = nullptr;

		// Default constructor
		FrameData() = default;

		// Construct from device
		FrameData(const vk::raii::Device &device)
			: fence(device, vk::FenceCreateInfo {vk::FenceCreateFlagBits::eSignaled}),
			present_completed(device, vk::SemaphoreCreateInfo {}),
			render_completed(device, vk::SemaphoreCreateInfo {}) {}
	};

	std::vector <FrameData>			frames;

	// Sync queue for between-frames operations
	SyncQueue			sync_queue;

	// Generate aux structures
	Context get_context();

	// Recreate swapchain
	void recreate_swapchain();
public:
	// Constructor
	BaseApp(const vk::raii::PhysicalDevice &,
			const std::string &,
			const vk::Extent2D &,
			const std::vector <const char *> &,
			const vk::AttachmentLoadOp & = vk::AttachmentLoadOp::eClear);

	// Requires a record function
	virtual void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer &) = 0;

	// Possbily override a termination function
	virtual void terminate();

	// Possibly override a resize function
	virtual void resize(const vk::Extent2D &);

	// Present frame
	void present();

	// Overload frame function
	virtual void frame() override;
};

}

#endif
