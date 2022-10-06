#ifndef BACKEND_H_
#define BACKEND_H_

// Standard headers
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

// Vulkan and GLFW
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <GLFW/glfw3.h>

#define KOBRA_VALIDATION_LAYERS
#define KOBRA_THROW_ERROR

// Engine headers
#include "common.hpp"
#include "core.hpp"
#include "logger.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

namespace kobra {

////////////////////////////
// Aliases and structures //
////////////////////////////

// Vulkan structures
struct Device {
	vk::raii::PhysicalDevice	*phdev = nullptr;
	vk::raii::Device		*device = nullptr;
};

struct Context {
	vk::raii::PhysicalDevice	*phdev = nullptr;
	vk::raii::Device		*device = nullptr;
	vk::raii::CommandPool		*command_pool = nullptr;
	vk::raii::DescriptorPool	*descriptor_pool = nullptr;
	vk::Extent2D			extent = {};
	vk::Format			swapchain_format = vk::Format::eUndefined;
	vk::Format			depth_format = vk::Format::eUndefined;

	Device dev() const {
		return Device {phdev, device};
	}
};

// Simpler Vulkan aliases
using CommandBuffer = vk::raii::CommandBuffer;
using Framebuffer = vk::raii::Framebuffer;
using Pipeline = vk::raii::Pipeline;
using PipelineLayout = vk::raii::PipelineLayout;
using RenderPass = vk::raii::RenderPass;

using SyncTask = std::pair <std::string, std::function <void ()>>;

class SyncQueue {
	std::queue <SyncTask>	_handlers;
	std::mutex			_mutex;
public:
	SyncQueue() = default;

	void push(const SyncTask &task) {
		std::lock_guard <std::mutex> lock(_mutex);
		_handlers.push(task);
	}

	void do_pop(bool log = true) {
		std::lock_guard <std::mutex> lock(_mutex);
		if (!_handlers.empty()) {
			if (log)
				std::cout << "SyncQueue: " << _handlers.front().first << std::endl;
			_handlers.front().second();
			_handlers.pop();
		}
	}

	size_t size() {
		std::lock_guard <std::mutex> lock(_mutex);
		return _handlers.size();
	}
};

// Render area, takes care of viewport and scissors
struct RenderArea {
	glm::vec2 min;
	glm::vec2 max;

	// TODO: scissor as well

	// Width and height
	uint32_t width() const {
		return (uint32_t)(max.x - min.x);
	}

	uint32_t height() const {
		return (uint32_t)(max.y - min.y);
	}

	// Number of pixels
	uint32_t pixels() const {
		return width() * height();
	}

	// Apply render area (viewport and scissor)
	void apply(const vk::raii::CommandBuffer &cmd, const vk::Extent2D &extent) const {
		float minx = min.x, miny = min.y;
		float maxx = max.x, maxy = max.y;

		if (min == glm::vec2 {-1, -1}) {
			minx = 0;
			miny = 0;
		}

		if (max == glm::vec2 {-1, -1}) {
			maxx = extent.width;
			maxy = extent.height;
		}

		cmd.setViewport(0,
			vk::Viewport {
				minx, miny,
				maxx - minx, maxy - miny,
				0.0f, 1.0f
			}
		);

		// Set scissor
		cmd.setScissor(0,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				extent
			}
		);
	}
};

//////////////////////
// Object factories //
//////////////////////

// Get (or create) the singleton context
const vk::raii::Context &get_vulkan_context();

// Get (or generate) the required extensions
const std::vector <const char *> &get_required_extensions();

// Initialize GLFW statically
void _initialize_glfw();

// Get (or create) the singleton instance
const vk::raii::Instance &get_vulkan_instance();

// Window type
struct Window {
	GLFWwindow	*handle;
	std::string	title;
	vk::Extent2D	extent;

	Window() = default;

	Window(const std::string &title, const vk::Extent2D &extent)
			: title(title), extent(extent) {
		_initialize_glfw();
		handle = glfwCreateWindow(
			extent.width, extent.height,
			title.c_str(),
			nullptr, nullptr
		);
	}

	~Window() {
		glfwDestroyWindow(handle);
	}

	// Set cursor mode
	void set_cursor_mode(int mode) {
		glfwSetInputMode(handle, GLFW_CURSOR, mode);
	}
};

// Create a surface given a window
vk::raii::SurfaceKHR make_surface(const Window &);

// Get all available physical devices
vk::raii::PhysicalDevices get_physical_devices();

// Check if a physical device supports a set of extensions
bool physical_device_able(const vk::raii::PhysicalDevice &,
		const std::vector <const char *> &);

// Pick physical device according to some criteria
vk::raii::PhysicalDevice pick_physical_device
	(const std::function <bool (const vk::raii::PhysicalDevice &)> &);

// Find graphics queue family
uint32_t find_graphics_queue_family(const vk::raii::PhysicalDevice &);

// Find present queue family
uint32_t find_present_queue_family(const vk::raii::PhysicalDevice &,
		const vk::raii::SurfaceKHR &);

// Coupling graphics and present queue families
struct QueueFamilyIndices {
	uint32_t graphics;
	uint32_t present;
};

// Get both graphics and present queue families
QueueFamilyIndices find_queue_families(const vk::raii::PhysicalDevice &,
		const vk::raii::SurfaceKHR &);

// Create logical device on an arbitrary queue
vk::raii::Device make_device(const vk::raii::PhysicalDevice &,
		const uint32_t, const uint32_t,
		const std::vector <const char *> & = {});

// Create a logical device
vk::raii::Device make_device(const vk::raii::PhysicalDevice &,
		const QueueFamilyIndices &,
		const std::vector <const char *> &);

// Load Vulkan extension functions
void load_vulkan_extensions(const vk::raii::Device &device);

// Find memory type
uint32_t find_memory_type(const vk::PhysicalDeviceMemoryProperties &,
		uint32_t, vk::MemoryPropertyFlags);

// Dump device properties
std::string dev_info(const vk::raii::PhysicalDevice &);

// Allocate device memory
vk::raii::DeviceMemory allocate_device_memory(const vk::raii::Device &,
		const vk::PhysicalDeviceMemoryProperties &,
		const vk::MemoryRequirements &,
		const vk::MemoryPropertyFlags &,
		bool = false);

// Create a command buffer
vk::raii::CommandBuffer make_command_buffer(const vk::raii::Device &,
		const vk::raii::CommandPool &);

// Pick a surface format
vk::SurfaceFormatKHR pick_surface_format(const vk::raii::PhysicalDevice &,
		const vk::raii::SurfaceKHR &);

// Pick a present mode
vk::PresentModeKHR pick_present_mode(const vk::raii::PhysicalDevice &,
		const vk::raii::SurfaceKHR &);

// Swapchain structure
struct Swapchain {
	vk::Format				format;
	vk::raii::SwapchainKHR			swapchain = nullptr;
	std::vector <VkImage>			images;
	std::vector <vk::raii::ImageView>	image_views;

	// Constructing a swapchain
	Swapchain(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::SurfaceKHR &surface,
			const vk::Extent2D &extent,
			const QueueFamilyIndices &indices,
			const vk::raii::SwapchainKHR *old_swapchain = nullptr) {
		// Pick a surface format
		auto surface_format = pick_surface_format(phdev, surface);
		format = surface_format.format;

		// Surface capabilities and extent
		vk::SurfaceCapabilitiesKHR capabilities =
				phdev.getSurfaceCapabilitiesKHR(*surface);

		// Set the surface extent
		vk::Extent2D swapchain_extent = extent;
		if (capabilities.currentExtent.width == std::numeric_limits <uint32_t>::max()) {
			swapchain_extent.width = std::clamp(
				swapchain_extent.width,
				capabilities.minImageExtent.width,
				capabilities.maxImageExtent.width
			);

			swapchain_extent.height = std::clamp(
				swapchain_extent.height,
				capabilities.minImageExtent.height,
				capabilities.maxImageExtent.height
			);
		} else {
			swapchain_extent = capabilities.currentExtent;
		}

		// Transform, etc
		vk::SurfaceTransformFlagBitsKHR transform =
			(capabilities.supportedTransforms &
			vk::SurfaceTransformFlagBitsKHR::eIdentity) ?
			vk::SurfaceTransformFlagBitsKHR::eIdentity :
			capabilities.currentTransform;

		// Composite alpha
		vk::CompositeAlphaFlagBitsKHR composite_alpha =
			(capabilities.supportedCompositeAlpha &
			vk::CompositeAlphaFlagBitsKHR::eOpaque) ?
			vk::CompositeAlphaFlagBitsKHR::eOpaque :
			vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;

		// Present mode
		vk::PresentModeKHR present_mode = pick_present_mode(phdev, surface);

		// Creation info
		vk::SwapchainCreateInfoKHR create_info {
			{},
			*surface,
			capabilities.minImageCount,
			format,
			surface_format.colorSpace,
			swapchain_extent,
			1,
			vk::ImageUsageFlagBits::eColorAttachment,
			vk::SharingMode::eExclusive,
			{},
			transform,
			composite_alpha,
			present_mode,
			true,
			(old_swapchain ? **old_swapchain : nullptr)
		};

		// In case graphics and present queues are different
		if (indices.graphics != indices.present) {
			create_info.imageSharingMode = vk::SharingMode::eConcurrent;
			create_info.queueFamilyIndexCount = 2;
			create_info.pQueueFamilyIndices = &indices.graphics;
		}

		// Create the swapchain
		swapchain = vk::raii::SwapchainKHR(device, create_info);

		// Get the swapchain images
		images = swapchain.getImages();

		// Create image views
		vk::ImageViewCreateInfo create_view_info {
			{}, {},
			vk::ImageViewType::e2D,
			format,
			{},
			vk::ImageSubresourceRange(
				vk::ImageAspectFlagBits::eColor,
				0, 1, 0, 1
			)
		};

		for (size_t i = 0; i < images.size(); i++) {
			create_view_info.image = images[i];
			image_views.emplace_back(device, create_view_info);
		}
	}

	// Null constructor
	Swapchain(std::nullptr_t) {}
};

// Transition image layout
void transition_image_layout(const vk::raii::CommandBuffer &,
		const vk::Image &,
		const vk::Format &,
		const vk::ImageLayout,
		const vk::ImageLayout);

// Image data wrapper
struct ImageData {
	vk::Format		format;
	vk::Extent2D		extent;
	vk::ImageTiling		tiling;
	vk::ImageUsageFlags	usage;
	vk::ImageLayout  	layout;
	vk::MemoryPropertyFlags	properties;
	vk::ImageAspectFlags	aspect_mask;
	vk::raii::Image		image = nullptr;
	vk::raii::DeviceMemory	memory = nullptr;
	vk::raii::ImageView	view = nullptr;

	// Constructors
	ImageData(const vk::raii::PhysicalDevice &phdev_,
			const vk::raii::Device &device_,
			const vk::Format &fmt_,
			const vk::Extent2D &ext_,
			vk::ImageTiling tiling_,
			vk::ImageUsageFlags usage_,
			vk::ImageLayout initial_layout_,
			vk::MemoryPropertyFlags memory_properties_,
			vk::ImageAspectFlags aspect_mask_,
			bool external_ = false)
			: format {fmt_},
			extent {ext_},
			tiling {tiling_},
			usage {usage_},
			layout {initial_layout_},
			properties {memory_properties_},
			aspect_mask {aspect_mask_} {
		vk::ImageCreateInfo image_info {
			vk::ImageCreateFlags {},
			vk::ImageType::e2D,
			format,
			vk::Extent3D {extent.width, extent.height, 1},
			1, 1,
			vk::SampleCountFlagBits::e1,
			tiling, usage | vk::ImageUsageFlagBits::eSampled,
			vk::SharingMode::eExclusive, {},
			vk::ImageLayout::eUndefined
		};

		VkExternalMemoryImageCreateInfo ext_info;
		ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
		ext_info.pNext = nullptr;

		image_info.pNext = nullptr;
		if (external_)
			image_info.pNext = &ext_info;

		image = vk::raii::Image {device_, image_info};

		memory = vk::raii::DeviceMemory {
			allocate_device_memory(
				device_, phdev_.getMemoryProperties(),
				image.getMemoryRequirements(),
				memory_properties_,
				external_
			)
		};

		image.bindMemory(*memory, 0);
		view = vk::raii::ImageView {
			device_,
			vk::ImageViewCreateInfo {
				{}, *image, vk::ImageViewType::e2D,
				format, {}, {aspect_mask, 0, 1, 0, 1}
			}
		};
	}

	ImageData(std::nullptr_t) {}

	// No copy
	ImageData(const ImageData &) = delete;
	ImageData &operator=(const ImageData &) = delete;

	// Move only
	ImageData(ImageData &&other)
		: format {other.format},
		extent {other.extent},
		tiling {other.tiling},
		usage {other.usage},
		layout {other.layout},
		properties {other.properties},
		aspect_mask {other.aspect_mask},
		image {std::move(other.image)},
		memory {std::move(other.memory)},
		view {std::move(other.view)} {}

	ImageData &operator=(ImageData &&other) {
		format = other.format;
		extent = other.extent;
		tiling = other.tiling;
		usage = other.usage;
		layout = other.layout;
		properties = other.properties;
		aspect_mask = other.aspect_mask;
		image = std::move(other.image);
		memory = std::move(other.memory);
		view = std::move(other.view);
		return *this;
	}

	// Get size in bytes
	vk::DeviceSize get_size() const {
		return image.getMemoryRequirements().size;
	}

	// Get memory handle
	int get_memory_handle(const vk::raii::Device &device) const {
		// TODO: need to ensure that the the VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION is enabled
		int fd = -1;
		VkMemoryGetFdInfoKHR fdInfo = { };
		fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
		fdInfo.memory = *memory;
		fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

		// TODO: load in backend.cpp
		auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(*device, "vkGetMemoryFdKHR");

		if (!func) {
			KOBRA_LOG_FUNC(Log::ERROR) << "vkGetMemoryFdKHR not found\n";
			return -1;
		}

		VkResult result = func(*device, &fdInfo, &fd);
		if (result != VK_SUCCESS) {
			KOBRA_LOG_FUNC(Log::ERROR) << "vkGetMemoryFdKHR failed\n";
			return -1;
		}

		return fd;
	}

	// Transition the image to a new layout
	void transition_layout(const vk::raii::CommandBuffer &cmd,
				const vk::ImageLayout &new_layout) {
		transition_image_layout(cmd, *image, format, layout, new_layout);
		layout = new_layout;
	}

	void transition_layout(const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const vk::ImageLayout &new_layout) {
		// Create temporary command buffer
		auto cmd = make_command_buffer(device, command_pool);
		cmd.begin({});

		// Transition the image to a new layout
		transition_layout(cmd, new_layout);
		cmd.end();

		// Submit the command buffer
		vk::raii::Queue queue {device, 0, 0};

		queue.submit(
			vk::SubmitInfo {
				0, nullptr, nullptr, 1, &*cmd
			},
			nullptr
		);

		// Wait for the command buffer to finish
		queue.waitIdle();
	}

	// Blank image
	static ImageData blank(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device) {
		return ImageData(phdev, device,
			vk::Format::eR8G8B8A8Unorm,
			vk::Extent2D(1, 1),
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::eUndefined,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor
		);
	}
};

// Buffer data wrapper
class BufferData {
	// Buffer permanent data
	const vk::raii::Device			*_device = nullptr;
	vk::PhysicalDeviceMemoryProperties	_memory_properties;
public:
	// Once-a-lifetime properties
	vk::DeviceSize				size;
	vk::BufferUsageFlags			flags;
	vk::MemoryPropertyFlags			memory_properties;

	vk::raii::Buffer			buffer = nullptr;
	vk::raii::DeviceMemory			memory = nullptr;

	// Constructors
	BufferData(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::DeviceSize &size,
			vk::BufferUsageFlags usage,
			vk::MemoryPropertyFlags memory_properties)
			: _device(&device),
			_memory_properties(phdev.getMemoryProperties()),
			size { size },
			flags { usage },
			memory_properties { memory_properties },

			buffer { device,
				vk::BufferCreateInfo {
					{}, size, usage,
				}
			},

			memory {
				allocate_device_memory(
					device, phdev.getMemoryProperties(),
					buffer.getMemoryRequirements(),
					memory_properties
				)
			} {
		buffer.bindMemory(*memory, 0);
	}

	BufferData(std::nullptr_t) {}

	// Upload data to buffer
	template <class T>
	bool upload(const std::vector <T> &data, const vk::DeviceSize &offset = 0, bool auto_resize = true) {
		static constexpr char prop_msg[] = "Buffer data must be host coherent and host visible";
		static constexpr char size_msg[] = "Buffer size is smaller than data size";

		// Resize status
		bool resized = false;

		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			prop_msg
		);

		KOBRA_ASSERT(data.size() * sizeof(T) <= size || auto_resize,
			std::string(size_msg) + " (data size = " + std::to_string(data.size() * sizeof(T))
			+ ", buffer size = " + std::to_string(size) + ")"
		);

		if (data.size() * sizeof(T) > size) {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

			KOBRA_LOG_FUNC(Log::WARN) << size_msg << " (size = " << size
				<< ", data size = " << data.size() * sizeof(T)
				<< ")" << std::endl;

#endif

			// Resize buffer
			resize(data.size() * sizeof(T));
			resized = true;
		}

		// Upload data
		char *ptr = (char *) memory.mapMemory(0, size);
		memcpy(ptr + offset, data.data(), data.size() * sizeof(T));
		memory.unmapMemory();

		return resized;
	}

	template <class T>
	bool upload(const T *const data, const vk::DeviceSize &size_, const vk::DeviceSize &offset = 0, bool auto_resize = true) {
		static constexpr char prop_msg[] = "Buffer data must be host coherent and host visible";
		static constexpr char size_msg[] = "Buffer size is smaller than data size";

		// Resize status
		bool resized = false;

		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			prop_msg
		);

		KOBRA_ASSERT(size_ <= size || auto_resize,
			std::string(size_msg) + " (data size = " + std::to_string(size_)
			+ ", buffer size = " + std::to_string(size) + ")"
		);

		if (size_ > size) {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

			KOBRA_LOG_FUNC(Log::WARN) << size_msg << " (size = " << size_
				<< ", buffer size = " << size << ")" << std::endl;

#endif

			// Resize buffer
			resize(size_);
			resized = true;
		}

		// Upload data
		char *ptr = (char *) memory.mapMemory(0, size);
		memcpy(ptr + offset, data, size_);
		memory.unmapMemory();

		return resized;
	}

	// Get buffer data
	template <class T>
	std::vector <T> download() const {
		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			"Buffer data must be host coherent and host visible"
		);

		// Download data
		std::vector <T> data(size / sizeof(T));
		void *ptr = memory.mapMemory(0, size);
		memcpy(data.data(), ptr, size);
		memory.unmapMemory();

		return data;
	}
	
	template <class T>
	void download(std::vector <T> &dat) const {
		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			"Buffer data must be host coherent and host visible"
		);

		// Download data
		size_t size_ = dat.size() * sizeof(T);
		void *ptr = memory.mapMemory(0, size);
		memcpy(dat.data(), ptr, size_);
		memory.unmapMemory();
	}

	// Resize buffer
	void resize(const vk::DeviceSize &size_) {
		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			"Buffer data must be host coherent and host visible"
		);

		// Resize buffer
		size = size_;

		buffer = vk::raii::Buffer {
			*_device,
			vk::BufferCreateInfo {
				{}, size, flags,
			}
		};

		memory = allocate_device_memory(
			*_device, _memory_properties,
			buffer.getMemoryRequirements(),
			memory_properties
		);

		// Bind memory
		buffer.bindMemory(*memory, 0);
	}
};

// Device address
vk::DeviceAddress buffer_addr(const vk::raii::Device &, const BufferData &);
vk::DeviceAddress acceleration_structure_addr(const vk::raii::Device &, const vk::raii::AccelerationStructureKHR &);

// Copy data to an image
void copy_data_to_image(const vk::raii::CommandBuffer &,
		const vk::raii::Buffer &,
		const vk::raii::Image &,
		const vk::Format &,
		uint32_t, uint32_t);

// Copy image data to staging buffer
void copy_image_to_buffer(const vk::raii::CommandBuffer &,
		const vk::raii::Image &,
		const vk::raii::Buffer &,
		const vk::Format &,
		uint32_t, uint32_t);

// Create ImageData object from byte data
ImageData make_image(const vk::raii::CommandBuffer &,
		const vk::raii::PhysicalDevice &,
		const vk::raii::Device &,
		BufferData &,
		uint32_t,
		uint32_t,
		byte *,
		const vk::Format &,
		vk::ImageTiling,
		vk::ImageUsageFlags,
		vk::MemoryPropertyFlags,
		vk::ImageAspectFlags,
		bool = false);

// Create ImageData object from a file
ImageData make_image(const vk::raii::CommandBuffer &,
		const vk::raii::PhysicalDevice &,
		const vk::raii::Device &,
		BufferData &,
		const std::string &,
		vk::ImageTiling,
		vk::ImageUsageFlags,
		vk::MemoryPropertyFlags,
		vk::ImageAspectFlags,
		bool = false);

// Create ImageData object from a file (without hastle of extra arguments)
// TODO: source file
inline ImageData make_image(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		uint32_t width,
		uint32_t height,
		byte *data,
		const vk::Format &format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags memory_properties,
		vk::ImageAspectFlags aspect_mask,
		bool external = false)
{
	// Image data to be returned
	ImageData img = nullptr;

	// Queue to submit commands to
	vk::raii::Queue queue {device, 0, 0};

	// Temporary command buffer
	auto temp_command_buffer = make_command_buffer(device, command_pool);

	// Record
	temp_command_buffer.begin(vk::CommandBufferBeginInfo {
		vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	});

	// Staging buffer
	BufferData staging_buffer = nullptr;

	// Create the texture
	img = std::move(make_image(temp_command_buffer,
		phdev, device, staging_buffer,
		width, height, data,
		format, tiling, usage,
		memory_properties, aspect_mask, external
	));

	// End
	temp_command_buffer.end();

	// Submit
	queue.submit(
		vk::SubmitInfo {
			0, nullptr, nullptr,
			1, &*temp_command_buffer
		},
		nullptr
	);

	// Wait
	queue.waitIdle();

	// Return image data
	return img;
}

// Create ImageData object from a file (without hastle of extra arguments)
// TODO: source file
inline ImageData make_image(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const std::string &filename,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags memory_properties,
		vk::ImageAspectFlags aspect_mask,
		bool external = false)
{
	// Image data to be returned
	ImageData img = nullptr;

	// Queue to submit commands to
	vk::raii::Queue queue {device, 0, 0};

	// Temporary command buffer
	auto temp_command_buffer = make_command_buffer(device, command_pool);

	// Record
	temp_command_buffer.begin(vk::CommandBufferBeginInfo {
		vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	});

	// Staging buffer
	BufferData staging_buffer = nullptr;

	// Create the texture
	img = std::move(make_image(temp_command_buffer,
		phdev, device, staging_buffer,
		filename, tiling, usage,
		memory_properties, aspect_mask, external
	));

	// End
	temp_command_buffer.end();

	// Submit
	queue.submit(
		vk::SubmitInfo {
			0, nullptr, nullptr, 1, &*temp_command_buffer
		},
		nullptr
	);

	// Wait
	queue.waitIdle();

	// Return image data
	return img;
}

// Create a sampler from an ImageData object
inline vk::raii::Sampler make_sampler(const vk::raii::Device &device, const ImageData &image)
{
	return vk::raii::Sampler {
		device,
		vk::SamplerCreateInfo {
			{},
			vk::Filter::eLinear,
			vk::Filter::eLinear,
			vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			0.0f,
			VK_FALSE,
			0.0f,
			VK_FALSE,
			vk::CompareOp::eNever,
			0.0f,
			0.0f,
			vk::BorderColor::eIntOpaqueBlack,
			VK_FALSE
		}
	};
}

// Depth buffer data wrapper
struct DepthBuffer : public ImageData {
	// Constructors
	DepthBuffer(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::Format &fmt,
			const vk::Extent2D &extent)
			: ImageData(phdev, device,
				fmt, extent,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eDepthStencilAttachment,
				vk::ImageLayout::eUndefined,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eDepth) {}

	// Null constructor
	DepthBuffer(std::nullptr_t) : ImageData(nullptr) {}
};

// Create a render pass
inline vk::raii::RenderPass make_render_pass(const vk::raii::Device &device,
		const std::vector <vk::Format> &attachment_formats,
		const std::vector <vk::AttachmentLoadOp> &attachment_load_ops,
		const vk::Format &depth_format,
		const vk::AttachmentLoadOp &depth_load_op)
{
	// List of attachments
	std::vector <vk::AttachmentDescription> attachments;
	std::vector <vk::AttachmentReference> attachment_refs;

	// Make sure there are enough load ops
	KOBRA_ASSERT(
		attachment_formats.size() == attachment_load_ops.size(),
		"Attachment format and load op count mismatch"
	);

	// Add render pass attachments
	for (int i = 0; i < attachment_formats.size(); i++) {
		auto format = attachment_formats[i];
		auto load_op = attachment_load_ops[i];

		// Make sure at least a valid color attachment is present
		KOBRA_ASSERT(
			format != vk::Format::eUndefined,
			"Color attachment format is undefined"
		);

		// Create color attachment
		vk::ImageLayout color_layout;
		if (load_op == vk::AttachmentLoadOp::eLoad)
			color_layout = vk::ImageLayout::ePresentSrcKHR;
		else
			color_layout = vk::ImageLayout::eUndefined;

		vk::AttachmentDescription color_attachment {
			{}, format,
			vk::SampleCountFlagBits::e1,
			load_op,
			vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			color_layout,
			vk::ImageLayout::ePresentSrcKHR
		};

		// Add color attachment to list
		attachments.push_back(color_attachment);

		// Add color attachment reference
		attachment_refs.push_back(
			vk::AttachmentReference {
				uint32_t(i), vk::ImageLayout::eColorAttachmentOptimal
			}
		);
	}

	// Create depth attachment
	if (depth_format != vk::Format::eUndefined) {
		vk::ImageLayout depth_layout;
		if (depth_load_op == vk::AttachmentLoadOp::eLoad)
			depth_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		else
			depth_layout = vk::ImageLayout::eUndefined;

		vk::AttachmentDescription depth_attachment {
			{}, depth_format,
			vk::SampleCountFlagBits::e1,
			depth_load_op,
			vk::AttachmentStoreOp::eDontCare,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			depth_layout,
			vk::ImageLayout::eDepthStencilAttachmentOptimal
		};

		// Add depth attachment to list
		attachments.push_back(depth_attachment);
	}

	/* Reference to attachments
	vk::AttachmentReference color_attachment_ref {
		0, vk::ImageLayout::eColorAttachmentOptimal
	}; */

	vk::AttachmentReference depth_attachment_ref {
		uint32_t(attachments.size() - 1),
		vk::ImageLayout::eDepthStencilAttachmentOptimal
	};

	// Subpasses
	vk::SubpassDescription subpass {
		{}, vk::PipelineBindPoint::eGraphics,
		{}, attachment_refs, {},
		(depth_format == vk::Format::eUndefined) ? nullptr : &depth_attachment_ref
	};

	// Creation info
	vk::RenderPassCreateInfo render_pass_info {
		{}, attachments, subpass
	};

	// Create render pass
	return vk::raii::RenderPass(device, render_pass_info);
}

// Create framebuffers
inline std::vector <vk::raii::Framebuffer> make_framebuffers
		(const vk::raii::Device &device,
		const vk::raii::RenderPass &render_pass,
		const std::vector <vk::raii::ImageView> &image_views,
		const vk::raii::ImageView *depth_image_view,
		const vk::Extent2D &extent)
{
	// Create attachments
	vk::ImageView attachments[2] {};
	attachments[1] = (depth_image_view == nullptr) ?
			vk::ImageView {} : **depth_image_view;

	// Create framebuffers
	vk::FramebufferCreateInfo framebuffer_info {
		{}, *render_pass,
		(depth_image_view == nullptr) ? 1u : 2u,
		attachments,
		extent.width, extent.height,1
	};

	std::vector <vk::raii::Framebuffer> framebuffers;

	framebuffers.reserve(image_views.size());
	for (const auto &image_view : image_views) {
		attachments[0] = *image_view;
		framebuffers.emplace_back(device, framebuffer_info);
	}

	return framebuffers;
}

// Create a shader module
inline vk::raii::ShaderModule make_shader_module(const vk::raii::Device &device,
		const std::string &path)
{
	// Read shader file
	auto spv = common::read_glob(path);

	// Create shader module
	return vk::raii::ShaderModule(device,
		vk::ShaderModuleCreateInfo {
			{}, spv
		}
	);
}

// Multiple shader modules
inline std::vector <vk::raii::ShaderModule> make_shader_modules
		(const vk::raii::Device &device,
		const std::vector <std::string> &paths)
{
	// Create shader modules
	std::vector <vk::raii::ShaderModule> modules;

	modules.reserve(paths.size());
	for (const auto &path : paths) {
		auto spv = common::read_glob(path);
		modules.emplace_back(device,
			vk::ShaderModuleCreateInfo {
				{}, spv
			}
		);
	}

	return modules;
}

// Create descriptor pool from a vector of pool sizes
inline vk::raii::DescriptorPool make_descriptor_pool(const vk::raii::Device &device,
		const std::vector <vk::DescriptorPoolSize> &pool_sizes)
{
	KOBRA_ASSERT(
		pool_sizes.size() > 0,
		"Descriptor pool size vector is empty"
	);

	uint32_t max_sets = 0;
	for (const auto &pool_size : pool_sizes)
		max_sets += pool_size.descriptorCount;

	KOBRA_ASSERT(
		max_sets > 0,
		"Descriptor pool size vector is empty"
	);

	// Create descriptor pool
	return vk::raii::DescriptorPool(device,
		vk::DescriptorPoolCreateInfo {
			{
				vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind
					| vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
			},
			max_sets, pool_sizes
		}
	);
}

// Create a descriptor set layout
using DSLB = std::tuple <uint32_t, vk::DescriptorType, uint32_t, vk::ShaderStageFlagBits>;

// TODO: is this function even required? 1:1 parameter mapping
inline vk::raii::DescriptorSetLayout make_descriptor_set_layout
		(const vk::raii::Device &device,
		const std::vector <DSLB> &bindings,
		const vk::DescriptorSetLayoutCreateFlags &flags = {})
{
	std::vector <vk::DescriptorSetLayoutBinding> layout_bindings(bindings.size());
	for (size_t i = 0; i < bindings.size(); ++i) {
		layout_bindings[i] = {
			std::get <0> (bindings[i]),
			std::get <1> (bindings[i]),
			std::get <2> (bindings[i]),
			std::get <3> (bindings[i])
		};
	}

	// Create descriptor set layout
	return vk::raii::DescriptorSetLayout(device,
		vk::DescriptorSetLayoutCreateInfo {
			flags, layout_bindings
		}
	);
}

// Binding to a descriptor set
inline void bind_ds(const vk::raii::Device &device,
		const vk::raii::DescriptorSet &dset,
		const vk::Sampler &sampler,
		const vk::ImageView &view,
		uint32_t binding)
{
	// Bind sampler to descriptor set
	auto dset_info = std::array <vk::DescriptorImageInfo, 1> {
		vk::DescriptorImageInfo {
			sampler, view,
			vk::ImageLayout::eShaderReadOnlyOptimal
		}
	};

	vk::WriteDescriptorSet dset_write {
		*dset,
		binding, 0,
		vk::DescriptorType::eCombinedImageSampler,
		dset_info
	};

	device.updateDescriptorSets(dset_write, nullptr);
}

inline void bind_ds(const vk::raii::Device &device,
		const vk::raii::DescriptorSet &dset,
		const vk::raii::Sampler &sampler,
		const ImageData &img,
		uint32_t binding)
{
	return bind_ds(device, dset, *sampler, *img.view, binding);
}

inline void bind_ds(const vk::raii::Device &device,
		const vk::raii::DescriptorSet &dset,
		const BufferData &buffer,
		const vk::DescriptorType &type,
		uint32_t binding)
{
	// Bind buffer to descriptor set
	auto dset_info = std::array <vk::DescriptorBufferInfo, 1> {
		vk::DescriptorBufferInfo {
			*buffer.buffer,
			0, buffer.size
		}
	};

	vk::WriteDescriptorSet dset_write {
		*dset,
		binding, 0,
		type,
		nullptr,
		dset_info
	};

	device.updateDescriptorSets(dset_write, nullptr);
}

// Info for creating a graphics pipeline
struct GraphicsPipelineInfo {
	const vk::raii::Device &device;
	const vk::raii::RenderPass &render_pass;

	vk::raii::ShaderModule vertex_shader;
	const vk::SpecializationInfo *vertex_specialization = nullptr;

	vk::raii::ShaderModule fragment_shader;
	const vk::SpecializationInfo *fragment_specialization = nullptr;

	vk::VertexInputBindingDescription vertex_binding;
	std::vector <vk::VertexInputAttributeDescription> vertex_attributes;

	const vk::raii::PipelineLayout &pipeline_layout;
	// const vk::raii::PipelineCache &pipeline_cache = nullptr;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eBack;
	vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;

	bool blend_enabled = true;
	bool depth_test = true;
	bool depth_write = true;
	bool no_bindings = false;

	uint32_t color_blend_attachments = 1;

	// Constructor
	GraphicsPipelineInfo(const vk::raii::Device &device,
			const vk::raii::RenderPass &render_pass,
			vk::raii::ShaderModule &&vertex_shader,
			const vk::SpecializationInfo *vertex_specialization,
			vk::raii::ShaderModule &&fragment_shader,
			const vk::SpecializationInfo *fragment_specialization,
			const vk::VertexInputBindingDescription &vertex_binding,
			const std::vector <vk::VertexInputAttributeDescription> &vertex_attributes,
			const vk::raii::PipelineLayout &pipeline_layout)
			: device(device),
			render_pass(render_pass),
			vertex_shader(std::move(vertex_shader)),
			vertex_specialization(vertex_specialization),
			fragment_shader(std::move(fragment_shader)),
			fragment_specialization(fragment_specialization),
			vertex_binding(vertex_binding),
			vertex_attributes(vertex_attributes),
			pipeline_layout(pipeline_layout) {}
};

vk::raii::Pipeline make_graphics_pipeline(const GraphicsPipelineInfo &);

}

#endif
