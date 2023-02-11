// More vulkan headers
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_format_traits.hpp>

// Engine headers
#include "../include/backend.hpp"
#include "../include/core.hpp"
#include "../include/image.hpp"

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
		extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
		extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

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
		logger("Vulkan", Log::ERROR) << pCallbackData->pMessage << std::endl;

#ifdef KOBRA_THROW_ERROR

		throw std::runtime_error("[Vulkan Validation Layer] "
			"An error occured in the validation layer");

#endif

#ifndef KOBRA_VALIDATION_ERROR_ONLY

	// Warnings
	} else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		logger("Vulkan", Log::WARN) << pCallbackData->pMessage << std::endl;


#ifdef KOBRA_THROW_WARNING

		throw std::runtime_error("[Vulkan Validation Layer] "
			"An warning occured in the validation layer");

#endif

	// Info
	} else {
		logger("Vulkan", Log::INFO) << pCallbackData->pMessage << std::endl;

#endif

	}

	return VK_FALSE;
}

// Initialize GLFW statically
void _initialize_glfw()
{
	static bool initialized = false;

	// Make sure Vulkan is initialized
	if (!initialized) {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		// glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
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

	instance = vk::raii::Instance {
		get_vulkan_context(),
		instance_info
	};

#ifdef KOBRA_VALIDATION_LAYERS

	static constexpr vk::DebugUtilsMessengerCreateInfoEXT debug_messenger_info {
		vk::DebugUtilsMessengerCreateFlagsEXT(),
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
			| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
			| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
		debug_logger
	};

	static vk::raii::DebugUtilsMessengerEXT debug_messenger
		{instance, debug_messenger_info};

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

	// Device features
	vk::PhysicalDeviceFeatures device_features;
	device_features.independentBlend = true;

	// Create the device
	vk::DeviceCreateInfo device_info {
		vk::DeviceCreateFlags(), queue_info,
		{}, extensions, &device_features, nullptr
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
	uint32_t type_index = find_memory_type(
		memory_properties,
		memory_requirements.memoryTypeBits,
		properties
	);

	vk::MemoryAllocateInfo alloc_info {
		memory_requirements.size, type_index
	};

	alloc_info.pNext = nullptr;
	if (external) {
		// TODO: make not heap allocated
		VkExportMemoryAllocateInfo *ext_info = new VkExportMemoryAllocateInfo;
		ext_info->sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
		ext_info->handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
		ext_info->pNext = nullptr;

		alloc_info.pNext = ext_info;
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

// Transition an image layout
void transition_image_layout(const vk::raii::CommandBuffer &cmd,
		const vk::Image &image,
		const vk::Format &format,
		const vk::ImageLayout old_layout,
		const vk::ImageLayout new_layout)
{
	// Source stage
	vk::AccessFlags src_access_mask = {};

	switch (old_layout) {
	case vk::ImageLayout::eColorAttachmentOptimal:
		src_access_mask = vk::AccessFlagBits::eColorAttachmentWrite;
		break;
	case vk::ImageLayout::ePresentSrcKHR:
		src_access_mask = vk::AccessFlagBits::eMemoryRead;
		break;
	case vk::ImageLayout::eTransferDstOptimal:
		src_access_mask = vk::AccessFlagBits::eTransferWrite;
		break;
	case vk::ImageLayout::eTransferSrcOptimal:
		src_access_mask = vk::AccessFlagBits::eTransferRead;
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
	case vk::ImageLayout::eColorAttachmentOptimal:
		source_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		break;
	case vk::ImageLayout::ePresentSrcKHR:
		source_stage = vk::PipelineStageFlagBits::eBottomOfPipe;
		break;
	case vk::ImageLayout::eTransferDstOptimal:
	case vk::ImageLayout::eTransferSrcOptimal:
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

// Copy image data to staging buffer
void copy_image_to_buffer(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Image &img,
		const vk::raii::Buffer &buffer,
		const vk::Format &format,
		uint32_t width, uint32_t height)
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

	// Copy image to buffer
	cmd.copyImageToBuffer(*img,
		vk::ImageLayout::eTransferSrcOptimal,
		*buffer, {region}
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
		// vk::ImageLayout::eUndefined,
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

	img.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

	// Copy the buffer to the image
	copy_data_to_image(cmd,
		buffer.buffer, img.image,
		img.format, width, height
	);

	// TODO: transition_image_layout should go to the detail namespace...
	img.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

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

	byte *data = load_texture(filename, width, height, channels);

	// Create the image
	vk::Extent2D extent {
		static_cast <uint32_t> (width),
		static_cast <uint32_t> (height)
	};

	// Get appropriate format
	ImageData img = ImageData(
		phdev, device,
		vk::Format::eR8G8B8A8Unorm, // TODO: what about other formats?
					    // generate list and compare for
					    // best match?
		extent,
		tiling,
		usage,
		// vk::ImageLayout::eUndefined,
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

	img.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

	// Copy the buffer to the image
	copy_data_to_image(cmd,
		buffer.buffer, img.image,
		img.format, width, height
	);

	img.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

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

// Create a graphics pipeline
// TODO: refactor to GraphicsPipeline::make
vk::raii::Pipeline make_graphics_pipeline(const GraphicsPipelineInfo &info)
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
		info.cull_mode, info.front_face, VK_FALSE, 0, 0, 0, 1
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
		vk::CompareOp::eLessOrEqual, false, false,
		stencil_info, stencil_info
	};

	// Color blend state
	std::vector <vk::PipelineColorBlendAttachmentState> color_blend_attachments;

	for (int i = 0; i < info.blend_attachments.size(); i++) {
		vk::PipelineColorBlendAttachmentState color_blend_attachment;

		// TODO: provide more options for configuring color blend attachments
		color_blend_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR
				| vk::ColorComponentFlagBits::eG
				| vk::ColorComponentFlagBits::eB;
				// | vk::ColorComponentFlagBits::eA;

		color_blend_attachment.blendEnable = info.blend_attachments[i];
		color_blend_attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		color_blend_attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		color_blend_attachment.colorBlendOp = vk::BlendOp::eAdd;
		color_blend_attachment.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
		color_blend_attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		color_blend_attachment.alphaBlendOp = vk::BlendOp::eSubtract;

		color_blend_attachments.push_back(color_blend_attachment);
	}

	vk::PipelineColorBlendStateCreateInfo color_blend_info {
		{}, VK_FALSE,
		vk::LogicOp::eCopy,
		uint32_t(info.blend_attachments.size()),
		color_blend_attachments.data(),
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
		nullptr,
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

}
