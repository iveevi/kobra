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
#include <string>
#include <utility>
#include <vector>
#include <execinfo.h>

// Vulkan and GLFW
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <GLFW/glfw3.h>

// Engine headers
#include "common.hpp"
#include "core.hpp"
#include "logger.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

namespace kobra {

////////////////////////////
// Aliases and structures //
////////////////////////////

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
};

//////////////////////
// Object factories //
//////////////////////

// Get (or create) the singleton context
const vk::raii::Context &get_vulkan_context();

// Get (or generate) the required extensions
inline const std::vector <const char *> &get_required_extensions()
{
	// Vector to return
	static std::vector <const char *> extensions;

	// Add if empty
	if (extensions.empty()) {
		// Add glfw extensions
		uint32_t glfw_extension_count;
		const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
		extensions.insert(extensions.end(), glfw_extensions, glfw_extensions + glfw_extension_count);

		// Additional extensions
		// TODO: debugging extension if debuggin enabled
		extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
		extensions.push_back("VK_KHR_get_physical_device_properties2");

#ifdef KOBRA_VALIDATION_LAYERS

		// Add validation layers
		extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

#endif

	}

	return extensions;
}

// Create debug messenger
static bool check_validation_layer_support(const std::vector <const char *> &validation_layers)
{
	// TODO: remove this initial part?
	uint32_t layer_count;
	vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

	std::vector <VkLayerProperties> available_layers(layer_count);

	vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
	for (const char *layer : validation_layers) {
		bool layerFound = false;

		for (const auto &properties : available_layers) {
			if (strcmp(layer, properties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
			return false;
	}

	return true;
}

static VkResult make_debug_messenger(const VkInstance &instance,
		const VkDebugUtilsMessengerCreateInfoEXT *create_info,
		const VkAllocationCallbacks *allocator,
		VkDebugUtilsMessengerEXT *debug_messenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(
			instance,
			"vkCreateDebugUtilsMessengerEXT"
		);

	if (func != nullptr) {
		return func(instance,
			create_info,
			allocator,
			debug_messenger
		);
	}

	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void destroy_debug_messenger(const VkInstance &instance,
		const VkDebugUtilsMessengerEXT &debug_messenger,
		const VkAllocationCallbacks *allocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(
			instance,
			"vkDestroyDebugUtilsMessengerEXT"
		);

	if (func != nullptr) {
		func(instance,
			debug_messenger,
			allocator
		);
	}
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_logger
		(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
		void *pUserData)
{
	// Errors
	if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		Logger::error() << "[Vulkan Validation Layer] "
			<< pCallbackData->pMessage << std::endl;

#ifdef KOBRA_THROW_ERROR

		throw std::runtime_error("[Vulkan Validation Layer] "
			"An error occured in the validation layer");

#elif defined KOBRA_PAUSE_ON_ERROR

		/* Print stack trace
		// TODO:
		void *array[10];
		size_t size;

		// get void*'s for all entries on the stack
		size = backtrace(array, 10);

		// print out all the frames to stderr
		fprintf(stderr, "Error:");
		backtrace_symbols_fd(array, size, stderr); */

		std::cout << "Enter a key to continue..." << std::endl;
		std::cin.get();

#endif

#ifndef KOBRA_VALIDATION_ERROR_ONLY

	// Warnings
	} else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		Logger::warn() << "[Vulkan Validation Layer] "
			<< pCallbackData->pMessage << std::endl;


#ifdef KOBRA_THROW_WARNING

		throw std::runtime_error("[Vulkan Validation Layer] "
			"An warning occured in the validation layer");

#endif

	// Info
	} else {

		Logger::notify() << "[Vulkan Validation Layer] "
			<< pCallbackData->pMessage << std::endl;

#endif

	}

	return VK_FALSE;
}

#ifdef KOBRA_VALIDATION_LAYERS

// RAII handle for debug messenger
struct DebugMessenger {
	VkInstance			instance = nullptr;
	VkDebugUtilsMessengerEXT	messenger = nullptr;

	// Destructor
	~DebugMessenger() {
		std::cout << "Destroying debug messenger" << std::endl;
		destroy_debug_messenger(instance, messenger, nullptr);
	}

	operator bool() const {
		return messenger != nullptr;
	}
};

#endif

// Initialize GLFW statically
inline void _initialize_glfw()
{
	static bool initialized = false;

	// Make sure Vulkan is initialized
	if (!initialized) {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		initialized = true;

		KOBRA_LOG_FUNC(ok) << "GLFW initialized\n";
	}
}

// Get (or create) the singleton instance
inline const vk::raii::Instance &get_vulkan_instance()
{
	static bool initialized = false;
	static vk::raii::Instance instance = nullptr;
	static vk::ApplicationInfo app_info {
		"Kobra",
		VK_MAKE_VERSION(1, 0, 0),
		"Kobra",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_3
	};

	// Skip if already initialized
	if (initialized)
		return instance;

	// Make sure GLFW is initialized
	_initialize_glfw();

#ifdef KOBRA_VALIDATION_LAYERS

	static const std::vector <const char *> validation_layers = {
		"VK_LAYER_KHRONOS_validation"
	};

	// Check if validation layers are available
	KOBRA_ASSERT(
		check_validation_layer_support(validation_layers),
		"Validation layers are not available"
	);

#endif
	static vk::InstanceCreateInfo instance_info {
		vk::InstanceCreateFlags(),
		&app_info,

#ifdef KOBRA_VALIDATION_LAYERS

		static_cast <uint32_t> (validation_layers.size()),
		validation_layers.data(),
#else

		0, nullptr,

#endif

		(uint32_t) get_required_extensions().size(),
		get_required_extensions().data()
	};

#ifdef KOBRA_VALIDATION_LAYERS

	static VkDebugUtilsMessengerEXT debug_messenger {};
	static bool debug_messenger_created = false;

	VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info {
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = debug_logger
	};

	instance_info.pNext = &debug_messenger_info;

#endif

	instance = vk::raii::Instance {
		get_vulkan_context(),
		instance_info
	};

#ifdef KOBRA_VALIDATION_LAYERS

	if (!debug_messenger_created) {
		VkResult result = make_debug_messenger(
			*instance,
			&debug_messenger_info,
			nullptr,
			&debug_messenger
		);

		KOBRA_ASSERT(
			result == VK_SUCCESS,
			"Failed to create debug messenger"
		);

		debug_messenger_created = true;
	}

#endif

	KOBRA_LOG_FUNC(ok) << "Vulkan instance created\n";
	initialized = true;

	return instance;
}

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
inline vk::raii::SurfaceKHR make_surface(const Window &window)
{
	// Create the surface
	VkSurfaceKHR surface;
	VkResult result = glfwCreateWindowSurface(
		*get_vulkan_instance(),
		window.handle,
		nullptr,
		&surface
	);

	KOBRA_ASSERT(result == VK_SUCCESS, "Failed to create surface");

	return vk::raii::SurfaceKHR {
		get_vulkan_instance(),
		surface
	};
}

// Get all available physical devices
inline vk::raii::PhysicalDevices get_physical_devices()
{
	return vk::raii::PhysicalDevices {
		get_vulkan_instance()
	};
}

// Check if a physical device supports a set of extensions
inline bool physical_device_able(const vk::raii::PhysicalDevice &phdev,
		const std::vector <const char *> &extensions)
{
	// Get the device extensions
	std::vector <vk::ExtensionProperties> device_extensions =
			phdev.enumerateDeviceExtensionProperties();

	// Check if all the extensions are supported
	for (const char *extension : extensions) {
		if (std::find_if(device_extensions.begin(), device_extensions.end(),
				[&extension](const vk::ExtensionProperties &prop) {
					return !strcmp(prop.extensionName, extension);
				}) == device_extensions.end()) {
			KOBRA_LOG_FUNC(warn) << "Extension \"" << extension
					<< "\" is not supported\n";
			return false;
		}
	}

	return true;
}

// Pick physical device according to some criteria
inline vk::raii::PhysicalDevice pick_physical_device
	(const std::function <bool (const vk::raii::PhysicalDevice &)> &predicate)
{
	// Get all the physical devices
	vk::raii::PhysicalDevices devices = get_physical_devices();

	// Find the first one that satisfies the predicate
	for (const vk::raii::PhysicalDevice &device : devices) {
		if (predicate(device))
			return device;
	}

	// If none found, throw an error
	KOBRA_LOG_FUNC(error) << "No physical device found\n";
	throw std::runtime_error("[Vulkan] No physical device found");
}

// Find graphics queue family
inline uint32_t find_graphics_queue_family(const vk::raii::PhysicalDevice &phdev)
{
	// Get the queue families
	std::vector <vk::QueueFamilyProperties> queue_families =
			phdev.getQueueFamilyProperties();

	// Find the first one that supports graphics
	for (uint32_t i = 0; i < queue_families.size(); i++) {
		if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics)
			return i;
	}

	// If none found, throw an error
	KOBRA_LOG_FUNC(error) << "No graphics queue family found\n";
	throw std::runtime_error("[Vulkan] No graphics queue family found");
}

// Find present queue family
inline uint32_t find_present_queue_family(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::SurfaceKHR &surface)
{
	// Get the queue families
	std::vector <vk::QueueFamilyProperties> queue_families =
			phdev.getQueueFamilyProperties();

	// Find the first one that supports presentation
	for (uint32_t i = 0; i < queue_families.size(); i++) {
		if (phdev.getSurfaceSupportKHR(i, *surface))
			return i;
	}

	// If none found, throw an error
	KOBRA_LOG_FUNC(error) << "No presentation queue family found\n";
	throw std::runtime_error("[Vulkan] No presentation queue family found");
}

// Coupling graphics and present queue families
struct QueueFamilyIndices {
	uint32_t graphics;
	uint32_t present;
};

// Get both graphics and present queue families
inline QueueFamilyIndices find_queue_families(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::SurfaceKHR &surface)
{
	return {
		find_graphics_queue_family(phdev),
		find_present_queue_family(phdev, surface)
	};
}

// Create logical device on an arbitrary queue
inline vk::raii::Device make_device(const vk::raii::PhysicalDevice &phdev,
		const uint32_t queue_family,
		const uint32_t queue_count,
		const std::vector <const char *> &extensions = {})
{
	// Queue priorities
	std::vector <float> queue_priorities(queue_count, 1.0f);

	// Create the device info
	vk::DeviceQueueCreateInfo queue_info {
		vk::DeviceQueueCreateFlags(),
		queue_family, queue_count,
		queue_priorities.data()
	};

	// Create the device
	vk::DeviceCreateInfo device_info {
		vk::DeviceCreateFlags(),
		queue_info,
		{}, extensions,
		nullptr, nullptr
	};

	return vk::raii::Device {
		phdev, device_info
	};
}

// Create a logical device
inline vk::raii::Device make_device(const vk::raii::PhysicalDevice &phdev,
		const QueueFamilyIndices &indices,
		const std::vector <const char *> &extensions)
{
	auto families = phdev.getQueueFamilyProperties();
	uint32_t count = families[indices.graphics].queueCount;
	return make_device(phdev, indices.graphics, count, extensions);
}

// Load Vulkan extension functions
void load_vulkan_extensions(const vk::raii::Device &device);

// Find memory type
inline uint32_t find_memory_type(const vk::PhysicalDeviceMemoryProperties &mem_props,
		uint32_t type_filter,
		vk::MemoryPropertyFlags properties)
{
	uint32_t type_index = uint32_t(~0);
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if ((type_filter & 1) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
			type_index = i;
			break;
		}

		type_filter >>= 1;
	}

	if (type_index == uint32_t(~0)) {
		KOBRA_LOG_FUNC(error) << "No memory type found\n";
		throw std::runtime_error("[Vulkan] No memory type found");
	}

	return type_index;
}

// Dump device properties
inline std::string dev_info(const vk::raii::PhysicalDevice &phdev)
{
	std::stringstream ss;

	ss << "Chosen device: " << phdev.getProperties().deviceName << std::endl;
	ss << "\tgraphics queue family: " << find_graphics_queue_family(phdev) << std::endl;

	auto queue_families = phdev.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queue_families.size(); i++) {
		auto flags = queue_families[i].queueFlags;
		ss << "\tqueue family [" << queue_families[i].queueCount << "] ";
		if (flags & vk::QueueFlagBits::eGraphics)
			ss << "Graphics ";
		if (flags & vk::QueueFlagBits::eCompute)
			ss << "Compute ";
		if (flags & vk::QueueFlagBits::eTransfer)
			ss << "Transfer ";
		if (flags & vk::QueueFlagBits::eSparseBinding)
			ss << "SparseBinding ";
		ss << std::endl;
	}

	return ss.str();
}

// Allocate device memory
inline vk::raii::DeviceMemory allocate_device_memory(const vk::raii::Device &device,
		const vk::PhysicalDeviceMemoryProperties &memory_properties,
		const vk::MemoryRequirements &memory_requirements,
		const vk::MemoryPropertyFlags &properties)
{
	uint32_t type_index = find_memory_type(memory_properties,
			memory_requirements.memoryTypeBits, properties);

	vk::MemoryAllocateInfo alloc_info {
		memory_requirements.size, type_index
	};

	return vk::raii::DeviceMemory {
		device, alloc_info
	};
}

// Create a command buffer
inline vk::raii::CommandBuffer make_command_buffer(const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool)
{
	vk::CommandBufferAllocateInfo alloc_info {
		*command_pool, vk::CommandBufferLevel::ePrimary, 1
	};

	return std::move(device.allocateCommandBuffers(alloc_info)[0]);
}

// Pick a surface format
inline vk::SurfaceFormatKHR pick_surface_format(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::SurfaceKHR &surface)
{
	// Constant formats
	static const std::vector <vk::SurfaceFormatKHR> target_formats = {
		{ vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear },
	};

	// Get the surface formats
	std::vector <vk::SurfaceFormatKHR> formats =
			phdev.getSurfaceFormatsKHR(*surface);

	// If there is only one format, return it
	if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
		return {
			vk::Format::eB8G8R8A8Unorm,
			vk::ColorSpaceKHR::eSrgbNonlinear
		};
	}

	// Find the first one that is supported
	for (const vk::SurfaceFormatKHR &format : formats) {
		if (std::find_if(target_formats.begin(), target_formats.end(),
				[&format](const vk::SurfaceFormatKHR &target) {
					return format.format == target.format &&
							format.colorSpace == target.colorSpace;
				}) != target_formats.end()) {
			return format;
		}
	}

	// If none found, throw an error
	KOBRA_LOG_FUNC(error) << "No supported surface format found\n";
	throw std::runtime_error("[Vulkan] No supported surface format found");
}

// Pick a present mode
inline vk::PresentModeKHR pick_present_mode(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::SurfaceKHR &surface)
{
	// Constant modes
	static const std::vector <vk::PresentModeKHR> target_modes = {
		vk::PresentModeKHR::eMailbox,
		vk::PresentModeKHR::eImmediate,
		vk::PresentModeKHR::eFifo
	};

	// Get the present modes
	std::vector <vk::PresentModeKHR> modes =
			phdev.getSurfacePresentModesKHR(*surface);

	// Prioritize mailbox mode
	if (std::find(modes.begin(), modes.end(), vk::PresentModeKHR::eMailbox) !=
			modes.end()) {
		return vk::PresentModeKHR::eMailbox;
	}

	// Find the first one that is supported
	for (const vk::PresentModeKHR &mode : modes) {
		if (std::find(target_modes.begin(), target_modes.end(), mode) !=
				target_modes.end()) {
			return mode;
		}
	}

	// If none found, throw an error
	KOBRA_LOG_FUNC(error) << "No supported present mode found\n";
	throw std::runtime_error("[Vulkan] No supported present mode found");
}

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
inline void transition_image_layout(const vk::raii::CommandBuffer &cmd,
		const vk::Image &image,
		const vk::Format &format,
		const vk::ImageLayout old_layout,
		const vk::ImageLayout new_layout)
{
	// Source stage
	vk::AccessFlags src_access_mask = {};

	switch (old_layout) {
	case vk::ImageLayout::eTransferDstOptimal:
		src_access_mask = vk::AccessFlagBits::eTransferWrite;
		break;
	case vk::ImageLayout::ePreinitialized:
		src_access_mask = vk::AccessFlagBits::eHostWrite;
		break;
	case vk::ImageLayout::eGeneral:
	case vk::ImageLayout::eUndefined:
		break;
	case vk::ImageLayout::eShaderReadOnlyOptimal:
		src_access_mask = vk::AccessFlagBits::eShaderRead;
		break;
	default:
		KOBRA_ASSERT(false, "Unsupported old layout " + vk::to_string(old_layout));
		break;
	}

	// Pipeline stage
	vk::PipelineStageFlags source_stage;
	switch (old_layout) {
	case vk::ImageLayout::eGeneral:
	case vk::ImageLayout::ePreinitialized:
		source_stage = vk::PipelineStageFlagBits::eHost;
		break;
	case vk::ImageLayout::eTransferDstOptimal:
		source_stage = vk::PipelineStageFlagBits::eTransfer;
		break;
	case vk::ImageLayout::eUndefined:
		source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
		break;
	case vk::ImageLayout::eShaderReadOnlyOptimal:
		source_stage = vk::PipelineStageFlagBits::eFragmentShader;
		break;
	default:
		KOBRA_ASSERT(false, "Unsupported old layout " + vk::to_string(old_layout));
		break;
	}

	// Destination stage
	vk::AccessFlags dst_access_mask = {};
	switch (new_layout) {
	case vk::ImageLayout::eColorAttachmentOptimal:
		dst_access_mask = vk::AccessFlagBits::eColorAttachmentWrite;
		break;
	case vk::ImageLayout::eDepthStencilAttachmentOptimal:
		dst_access_mask = vk::AccessFlagBits::eDepthStencilAttachmentRead
			| vk::AccessFlagBits::eDepthStencilAttachmentWrite;
		break;
	case vk::ImageLayout::eGeneral:
	case vk::ImageLayout::ePresentSrcKHR:
		break;
	case vk::ImageLayout::eShaderReadOnlyOptimal:
		dst_access_mask = vk::AccessFlagBits::eShaderRead;
		break;
	case vk::ImageLayout::eTransferSrcOptimal:
		dst_access_mask = vk::AccessFlagBits::eTransferRead;
		break;
	case vk::ImageLayout::eTransferDstOptimal:
		dst_access_mask = vk::AccessFlagBits::eTransferWrite;
		break;
	default:
		KOBRA_ASSERT(false, "Unsupported new layout " + vk::to_string(new_layout));
		break;
	}

	// Destination stage
	vk::PipelineStageFlags destination_stage;
	switch (new_layout) {
	case vk::ImageLayout::eColorAttachmentOptimal:
		destination_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput; break;
	case vk::ImageLayout::eDepthStencilAttachmentOptimal:
		destination_stage = vk::PipelineStageFlagBits::eEarlyFragmentTests; break;
	case vk::ImageLayout::eGeneral:
		destination_stage = vk::PipelineStageFlagBits::eHost; break;
	case vk::ImageLayout::ePresentSrcKHR:
		destination_stage = vk::PipelineStageFlagBits::eBottomOfPipe; break;
	case vk::ImageLayout::eShaderReadOnlyOptimal:
		destination_stage = vk::PipelineStageFlagBits::eFragmentShader; break;
	case vk::ImageLayout::eTransferDstOptimal:
	case vk::ImageLayout::eTransferSrcOptimal:
		destination_stage = vk::PipelineStageFlagBits::eTransfer; break;
	default:
		KOBRA_ASSERT(false, "Unsupported new layout " + vk::to_string(new_layout));
		break;
	}

	// Aspect mask
	vk::ImageAspectFlags aspect_mask;
	if (new_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
		aspect_mask = vk::ImageAspectFlagBits::eDepth;
		if (format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint)
			aspect_mask |= vk::ImageAspectFlagBits::eStencil;
	} else {
		aspect_mask = vk::ImageAspectFlagBits::eColor;
	}

	// Create the barrier
	vk::ImageSubresourceRange image_subresource_range {
		aspect_mask,
			0, 1, 0, 1
	};

	vk::ImageMemoryBarrier barrier {
		src_access_mask, dst_access_mask,
		old_layout, new_layout,
		VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
		image, image_subresource_range
	};

	// Add the barrier
	return cmd.pipelineBarrier(source_stage, destination_stage, {}, {}, {}, barrier);
}

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
			vk::ImageAspectFlags aspect_mask_)
			: format {fmt_},
			extent {ext_},
			tiling {tiling_},
			usage {usage_},
			layout {initial_layout_},
			properties {memory_properties_},
			aspect_mask {aspect_mask_},
			image {device_,
				{
					vk::ImageCreateFlags(),
					vk::ImageType::e2D,
					format,
					vk::Extent3D( extent, 1 ),
					1,
					1,
					vk::SampleCountFlagBits::e1,
					tiling,
					usage | vk::ImageUsageFlagBits::eSampled,
					vk::SharingMode::eExclusive,
					{},
					layout
				}
			},

			memory {
				allocate_device_memory(
					device_, phdev_.getMemoryProperties(),
					image.getMemoryRequirements(),
					memory_properties_
				)
			} {
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

	/* Create a copy of the image data
	ImageData copy(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool) const {
		ImageData copy {
			phdev, device,
			format, extent, tiling,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageLayout::ePreinitialized,
			properties,
			aspect_mask
		};

		// Create a temporary command buffer
		auto cmd = make_command_buffer(device, command_pool);

		// Transition layuot, then copy image data
		{
			// Start recording the command buffer
			cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

			// Transition for copying
			transition_image_layout(cmd, *image, format, layout, vk::ImageLayout::eTransferSrcOptimal);

			vk::ImageCopy copy_region {
				{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
				{0, 0, 0},
				{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
				{0, 0, 0},
				{extent.width, extent.height, 1}
			};

			cmd.copyImage(
				*image, vk::ImageLayout::eTransferSrcOptimal,
				*copy.image, vk::ImageLayout::eTransferDstOptimal,
				copy_region
			);

			// Transition to original layout
			// transition_image_layout(cmd, *image, format, vk::ImageLayout::eTransferSrcOptimal, layout);

			// End recording the command buffer
			cmd.end();
		}

		// Queue to submit commands to
		vk::raii::Queue queue {device, 0, 0};

		// Submit the command buffer
		queue.submit(
			vk::SubmitInfo {
				0, nullptr, nullptr, 1, &*cmd
			},
			nullptr
		);

		// Wait for the command buffer to finish
		queue.waitIdle();

		return copy;
	} */

	// Blank image
	static ImageData blank(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device) {
		return ImageData(phdev, device,
			vk::Format::eR8G8B8A8Unorm,
			vk::Extent2D(1, 1),
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::ePreinitialized,
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

		bool smaller = (data.size() * sizeof(T) > size);

		KOBRA_ASSERT(smaller || auto_resize, size_msg);
		if (smaller) {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

			KOBRA_LOG_FUNC(warn) << size_msg << " (size = " << size
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

		bool smaller = (size_ > size);
		KOBRA_ASSERT(smaller || auto_resize, size_msg);

		if (smaller) {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

			KOBRA_LOG_FUNC(warn) << size_msg << " (size = " << size_
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
		vk::ImageAspectFlags);

// Create ImageData object from a file
ImageData make_image(const vk::raii::CommandBuffer &,
		const vk::raii::PhysicalDevice &,
		const vk::raii::Device &,
		BufferData &,
		const std::string &,
		vk::ImageTiling,
		vk::ImageUsageFlags,
		vk::MemoryPropertyFlags,
		vk::ImageAspectFlags);

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
		vk::ImageAspectFlags aspect_mask)
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
		memory_properties, aspect_mask
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

// Create ImageData object from a file (without hastle of extra arguments)
// TODO: source file
inline ImageData make_image(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const std::string &filename,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags memory_properties,
		vk::ImageAspectFlags aspect_mask)
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
		memory_properties, aspect_mask
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
		const vk::Format &format,
		const vk::Format &depth_format,
		const vk::AttachmentLoadOp &load_op = vk::AttachmentLoadOp::eClear,
		const vk::ImageLayout &initial_layout = vk::ImageLayout::ePresentSrcKHR)
{
	// List of attachments
	std::vector <vk::AttachmentDescription> attachments;

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

	// Create depth attachment
	if (depth_format != vk::Format::eUndefined) {
		vk::ImageLayout depth_layout;
		if (load_op == vk::AttachmentLoadOp::eLoad)
			depth_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		else
			depth_layout = vk::ImageLayout::eUndefined;

		vk::AttachmentDescription depth_attachment {
			{}, depth_format,
			vk::SampleCountFlagBits::e1,
			load_op,
			vk::AttachmentStoreOp::eDontCare,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			depth_layout,
			vk::ImageLayout::eDepthStencilAttachmentOptimal
		};

		// Add depth attachment to list
		attachments.push_back(depth_attachment);
	}

	// Reference to attachments
	vk::AttachmentReference color_attachment_ref {
		0, vk::ImageLayout::eColorAttachmentOptimal
	};

	vk::AttachmentReference depth_attachment_ref {
		1, vk::ImageLayout::eDepthStencilAttachmentOptimal
	};

	// Subpasses
	vk::SubpassDescription subpass {
		{}, vk::PipelineBindPoint::eGraphics,
		{}, color_attachment_ref,
		{},
		(depth_format == vk::Format::eUndefined) ? nullptr : &depth_attachment_ref
	};

	// Creation info
	vk::RenderPassCreateInfo render_pass_info {
		{}, attachments,
		subpass
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
		extent.width, extent.height, 1
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

// Create a graphics pipeline
struct GraphicsPipelineInfo {
	const vk::raii::Device &device;
	const vk::raii::RenderPass &render_pass;

	vk::raii::ShaderModule vertex_shader;
	const vk::SpecializationInfo *vertex_specialization = nullptr;

	vk::raii::ShaderModule fragment_shader;
	const vk::SpecializationInfo *fragment_specialization = nullptr;

	bool no_bindings = false;
	vk::VertexInputBindingDescription vertex_binding;
	std::vector <vk::VertexInputAttributeDescription> vertex_attributes;

	const vk::raii::PipelineLayout &pipeline_layout;
	const vk::raii::PipelineCache &pipeline_cache;

	bool depth_test;
	bool depth_write;
};

inline vk::raii::Pipeline make_graphics_pipeline(const GraphicsPipelineInfo &info)
{
	// Shader stages
	std::array <vk::PipelineShaderStageCreateInfo, 2> shader_stages {
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eVertex,
			*info.vertex_shader, "main",
			info.vertex_specialization
		},
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eFragment,
			*info.fragment_shader, "main",
			info.fragment_specialization
		}
	};

	/* std::cout << "Vertex attribute formats:" << info.vertex_attributes.size() << std::endl;
	for (const auto &attr : info.vertex_attributes) {
		std::cout << "\t" << attr.binding << " " << attr.location << " "
			<< vk::to_string(attr.format) << " " << attr.offset << std::endl;
	} */

	// Vertex input state
	vk::PipelineVertexInputStateCreateInfo vertex_input_info {
		{}, !info.no_bindings, &info.vertex_binding,
		(uint32_t) info.vertex_attributes.size(),
		info.vertex_attributes.data()
	};

	// Input assembly state
	vk::PipelineInputAssemblyStateCreateInfo input_assembly_info {
		{}, vk::PrimitiveTopology::eTriangleList, VK_FALSE
	};

	// Viewport state
	vk::PipelineViewportStateCreateInfo viewport_state_info {
		{}, 1, nullptr, 1, nullptr
	};

	// Rasterization state
	vk::PipelineRasterizationStateCreateInfo rasterization_info {
		{}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
		vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
		VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f
	};

	// Multisample state
	vk::PipelineMultisampleStateCreateInfo multisample_info {
		{}, vk::SampleCountFlagBits::e1, VK_FALSE, 0.0f, nullptr,
		VK_FALSE, VK_FALSE
	};

	// Depth stencil state
	vk::StencilOpState stencil_info {
		vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep,
		vk::CompareOp::eAlways, 0, 0, 0
	};

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {
		{}, info.depth_test, info.depth_write,
		vk::CompareOp::eLess, false, false,
		stencil_info, stencil_info
	};

	// Color blend state
	vk::PipelineColorBlendAttachmentState color_blend_attachment {
		VK_FALSE, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
		vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
		vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo color_blend_info {
		{}, VK_FALSE, vk::LogicOp::eCopy, 1, &color_blend_attachment,
		{ 0.0f, 0.0f, 0.0f, 0.0f }
	};

	// Dynamic state
	std::array <vk::DynamicState, 2> dynamic_states {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};

	vk::PipelineDynamicStateCreateInfo dynamic_state_info {
		{}, (uint32_t) dynamic_states.size(), dynamic_states.data()
	};

	// Pipeline
	return vk::raii::Pipeline {
		info.device,
		info.pipeline_cache,
		vk::GraphicsPipelineCreateInfo {
			{}, shader_stages,
			&vertex_input_info,
			&input_assembly_info,
			nullptr,
			&viewport_state_info,
			&rasterization_info,
			&multisample_info,
			&depth_stencil_info,
			&color_blend_info,
			&dynamic_state_info,
			*info.pipeline_layout,
			*info.render_pass
		}
	};
}

// TODO: compiling GLSL into SPIRV in runtime

}

#endif
