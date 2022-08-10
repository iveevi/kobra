// More vulkan headers
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_format_traits.hpp>

// STBI headrs
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "../include/backend.hpp"
#include "../include/core.hpp"

namespace kobra {

// Get (or create) the singleton context
const vk::raii::Context &get_vulkan_context()
{
	// Global context
	static vk::raii::Context context;
	return context;
}

// Get (or generate) the required extensions
const std::vector <const char *> &get_required_extensions()
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
		// extensions.push_back("VK_KHR_get_physical_device_properties2");
		extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
		// extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
		// extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);

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
void _initialize_glfw()
{
	static bool initialized = false;

	// Make sure Vulkan is initialized
	if (!initialized) {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		initialized = true;

		KOBRA_LOG_FUNC(Log::OK) << "GLFW initialized\n";
	}
}

// Get (or create) the singleton instance
const vk::raii::Instance &get_vulkan_instance()
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

	KOBRA_LOG_FUNC(Log::OK) << "Vulkan instance created\n";
	initialized = true;

	return instance;
}

// Create a surface given a window
vk::raii::SurfaceKHR make_surface(const Window &window)
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
vk::raii::PhysicalDevices get_physical_devices()
{
	return vk::raii::PhysicalDevices {
		get_vulkan_instance()
	};
}

// Check if a physical device supports a set of extensions
bool physical_device_able(const vk::raii::PhysicalDevice &phdev,
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
			KOBRA_LOG_FUNC(Log::WARN) << "Extension \"" << extension
					<< "\" is not supported\n";
			return false;
		}
	}

	return true;
}

// Pick physical device according to some criteria
vk::raii::PhysicalDevice pick_physical_device
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
	KOBRA_LOG_FUNC(Log::ERROR) << "No physical device found\n";
	throw std::runtime_error("[Vulkan] No physical device found");
}

// Find graphics queue family
uint32_t find_graphics_queue_family(const vk::raii::PhysicalDevice &phdev)
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
	KOBRA_LOG_FUNC(Log::ERROR) << "No graphics queue family found\n";
	throw std::runtime_error("[Vulkan] No graphics queue family found");
}

// Find present queue family
uint32_t find_present_queue_family(const vk::raii::PhysicalDevice &phdev,
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
	KOBRA_LOG_FUNC(Log::ERROR) << "No presentation queue family found\n";
	throw std::runtime_error("[Vulkan] No presentation queue family found");
}

// Get both graphics and present queue families
QueueFamilyIndices find_queue_families(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::SurfaceKHR &surface)
{
	return {
		find_graphics_queue_family(phdev),
		find_present_queue_family(phdev, surface)
	};
}

// Create logical device on an arbitrary queue
vk::raii::Device make_device(const vk::raii::PhysicalDevice &phdev,
		const uint32_t queue_family,
		const uint32_t queue_count,
		const std::vector <const char *> &extensions)
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
vk::raii::Device make_device(const vk::raii::PhysicalDevice &phdev,
		const QueueFamilyIndices &indices,
		const std::vector <const char *> &extensions)
{
	auto families = phdev.getQueueFamilyProperties();
	uint32_t count = families[indices.graphics].queueCount;
	return make_device(phdev, indices.graphics, count, extensions);
}

// Find memory type
uint32_t find_memory_type(const vk::PhysicalDeviceMemoryProperties &mem_props,
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
		KOBRA_LOG_FUNC(Log::ERROR) << "No memory type found\n";
		throw std::runtime_error("[Vulkan] No memory type found");
	}

	return type_index;
}

// Dump device properties
std::string dev_info(const vk::raii::PhysicalDevice &phdev)
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
// TODO: external should be a struct?
vk::raii::DeviceMemory allocate_device_memory(const vk::raii::Device &device,
		const vk::PhysicalDeviceMemoryProperties &memory_properties,
		const vk::MemoryRequirements &memory_requirements,
		const vk::MemoryPropertyFlags &properties,
		bool external)
{
	uint32_t type_index = find_memory_type(memory_properties,
			memory_requirements.memoryTypeBits, properties);

	vk::MemoryAllocateInfo alloc_info {
		memory_requirements.size, type_index
	};

	if (external) {
		vk::ExportMemoryAllocateInfo ext_info {
			vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd
		};

		alloc_info.pNext = &ext_info;
	}

	return vk::raii::DeviceMemory {
		device, alloc_info
	};
}

// Create a command buffer
vk::raii::CommandBuffer make_command_buffer(const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool)
{
	vk::CommandBufferAllocateInfo alloc_info {
		*command_pool, vk::CommandBufferLevel::ePrimary, 1
	};

	return std::move(device.allocateCommandBuffers(alloc_info)[0]);
}

// Pick a surface format
vk::SurfaceFormatKHR pick_surface_format(const vk::raii::PhysicalDevice &phdev,
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
	KOBRA_LOG_FUNC(Log::ERROR) << "No supported surface format found\n";
	throw std::runtime_error("[Vulkan] No supported surface format found");
}

// Pick a present mode
vk::PresentModeKHR pick_present_mode(const vk::raii::PhysicalDevice &phdev,
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
	KOBRA_LOG_FUNC(Log::ERROR) << "No supported present mode found\n";
	throw std::runtime_error("[Vulkan] No supported present mode found");
}

// Copy a buffer to an image
void copy_data_to_image(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Buffer &buffer,
		const vk::raii::Image &image,
		const vk::Format &format,
		uint32_t width,
		uint32_t height)
{
	// Image subresource
	vk::ImageSubresourceLayers subresource {
		vk::ImageAspectFlagBits::eColor,
		0, 0, 1
	};

	// Copy region
	vk::BufferImageCopy region = vk::BufferImageCopy()
		.setBufferOffset(0)
		.setBufferRowLength(width)
		.setBufferImageHeight(height)
		.setImageSubresource(subresource)
		.setImageOffset({ 0, 0, 0 })
		.setImageExtent({ width, height, 1 });

	// Copy buffer to image
	cmd.copyBufferToImage(*buffer, *image,
		vk::ImageLayout::eTransferDstOptimal,
		{region}
	);
}

// Create ImageData object from byte data
ImageData make_image(const vk::raii::CommandBuffer &cmd,
		const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		BufferData &buffer,
		uint32_t width,
		uint32_t height,
		byte *data,
		const vk::Format &format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags memory_properties,
		vk::ImageAspectFlags aspect_mask,
		bool external)
{
	// Create the image
	vk::Extent2D extent {
		static_cast <uint32_t> (width),
		static_cast <uint32_t> (height)
	};

	ImageData img = ImageData(
		phdev, device,
		format,
		extent,
		tiling,
		usage,
		vk::ImageLayout::ePreinitialized,
		memory_properties,
		aspect_mask, external
	);

	// Copy the image data into a staging buffer
	vk::DeviceSize size = width * height * vk::blockSize(img.format);

	buffer = BufferData(
		phdev, device,
		size,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Copy the data
	buffer.upload(data, size);

	// First transition the image to the transfer destination layout
	transition_image_layout(cmd,
		*img.image, img.format,
		vk::ImageLayout::ePreinitialized,
		vk::ImageLayout::eTransferDstOptimal
	);

	// Copy the buffer to the image
	copy_data_to_image(cmd,
		buffer.buffer, img.image,
		img.format, width, height
	);

	// Transition the image to the shader read layout
	transition_image_layout(cmd,
		*img.image, img.format,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal
	);

	return img;
}

// Create ImageData object from a file
// TODO: delegate to the above function
ImageData make_image(const vk::raii::CommandBuffer &cmd,
		const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		BufferData &buffer,
		const std::string &filename,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags memory_properties,
		vk::ImageAspectFlags aspect_mask,
		bool external)
{
	// Check if the file exists
	KOBRA_ASSERT(common::file_exists(filename), "File not found: " + filename);

	// Load the image
	int width;
	int height;
	int channels;

	stbi_set_flip_vertically_on_load(true);
	byte *data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
	KOBRA_ASSERT(data, "Failed to load texture image");

	KOBRA_LOG_FUNC(Log::INFO) << "Loaded image: " << filename << ": width="
			<< width << ", height=" << height << ", channels=" << channels << "\n";

	/* uint32_t *pixels = (uint32_t *)data;
	std::cout << "Image dump: " << filename << std::endl;
	uint32_t r = 0, g = 0, b = 0, a = 0;
	for (int i = 0; i < 20; i++) {
		uint32_t v = pixels[i];
		r = (v >> 24) & 0xff;
		g = (v >> 16) & 0xff;
		b = (v >> 8) & 0xff;
		a = v & 0xff;
		std::cout << "(" << r << ", " << g << ", " << b << ", " << a << ")\n";
	} */

	// Create the image
	vk::Extent2D extent {
		static_cast <uint32_t> (width),
		static_cast <uint32_t> (height)
	};

	ImageData img = ImageData(
		phdev, device,
		vk::Format::eR8G8B8A8Unorm,
		extent,
		tiling,
		usage,
		vk::ImageLayout::ePreinitialized,
		memory_properties,
		aspect_mask, external
	);

	// Copy the image data into a staging buffer
	vk::DeviceSize size = width * height * vk::blockSize(img.format);

	std::cout << "Buffer (staging) size = " << size << std::endl;

	buffer = BufferData(
		phdev, device,
		size,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Copy the data
	buffer.upload(data, size);

	// First transition the image to the transfer destination layout
	transition_image_layout(cmd,
		*img.image, img.format,
		vk::ImageLayout::ePreinitialized,
		vk::ImageLayout::eTransferDstOptimal
	);

	// Copy the buffer to the image
	copy_data_to_image(cmd,
		buffer.buffer, img.image,
		img.format, width, height
	);

	// Transition the image to the shader read layout
	transition_image_layout(cmd,
		*img.image, img.format,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal
	);

	return img;
}

// Buffer addresses
vk::DeviceAddress buffer_addr(const vk::raii::Device &device, const BufferData &bd)
{
	VkBufferDeviceAddressInfo info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = *bd.buffer
	};

	return vkGetBufferDeviceAddressKHR(*device, &info);
}

// Acceleration structure address
vk::DeviceAddress acceleration_structure_addr(const vk::raii::Device &device, const vk::raii::AccelerationStructureKHR &as)
{
	VkAccelerationStructureDeviceAddressInfoKHR info {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.accelerationStructure = *as
	};

	return vkGetAccelerationStructureDeviceAddressKHR(*device, &info);
}

}
