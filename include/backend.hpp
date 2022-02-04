#ifndef BACKEND_H_
#define BACKEND_H_

// Standard headers
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// TODO: remove the glad diretcory/deps
// GLFW and Vulkan
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// ImGui headers
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

// Engine headers
#include "logger.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector <const char *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector <const char *> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

class Vulkan {
public:
	///////////////////////
	// Public structures //
	///////////////////////
	
	// Buffer information
	struct Buffer {
		VkBuffer	buffer;
		VkDeviceMemory	memory;
		VkDeviceSize	size;
		VkDeviceSize	offset;
		void *		mapped;
	};

	// Aliases
	using Glob = std::vector <char>;

	using CommandBufferMaker = void (*)(Vulkan *, size_t);
	using DeletionTask = std::function <void (Vulkan *)>;	// TODO: Is this Vulkan object needed?

	using DS = VkDescriptorSet;
	using DSLayout = VkDescriptorSetLayout;

	/////////////////////
	// Exposed members //
	/////////////////////
	
	// GLFW window
	GLFWwindow *window;

	// Vulkan basic context
	VkInstance instance;
	
	VkSurfaceKHR surface;

	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkDevice device;

	// Queues
	VkQueue graphics_queue;
	VkQueue present_queue;

	// Swapchain variables
	VkSwapchainKHR			swch;
	VkExtent2D			swch_extent;
	VkFormat			swch_image_format;
	std::vector <VkFramebuffer>	swch_framebuffers;
	std::vector <VkImage>		swch_images;
	std::vector <VkImageView>	swch_image_views;

	// Rendering variables
	VkRenderPass			render_pass;

	VkCommandPool			command_pool;

	std::vector <VkCommandBuffer>	command_buffers;
	
	std::vector <VkFence>		in_flight_fences;
	std::vector <VkFence>		images_in_flight;

	std::vector <VkSemaphore>	image_available_semaphores;
	std::vector <VkSemaphore>	render_finished_semaphores;

	// Miscellaneous
	VkAllocationCallbacks *		allocator = nullptr;
	
	// Descriptor pool
	VkDescriptorPool		descriptor_pool;

	// Descriptor sets
	DSLayout			ds_layout;
	std::vector <DSLayout>		descriptor_set_layouts;
	std::vector <DS>		descriptor_sets;
private:

	// Internal structures
	// TODO: rearrange by task topic
	struct QueueFamilyIndices {
		std::optional <uint32_t> graphics;
		std::optional <uint32_t> present;

		// As a boolean status
		operator bool() {
			return graphics.has_value()
				&& present.has_value();
		}
	};

	struct SwapchainSupport {
		VkSurfaceCapabilitiesKHR		capabilities;
		std::vector <VkSurfaceFormatKHR>	formats;
		std::vector <VkPresentModeKHR>		presentModes;
	};
	
	//////////////////////////////
	// Private member variables //
	//////////////////////////////
	
	VkDebugUtilsMessengerEXT _debug_messenger;

	// Frame and window variables
	size_t				current_frame = 0;
	bool				framebuffer_resized = false;

	// Storage of allocated structures
	// for easier cleanup
	std::vector <Buffer>		_buffers;

	// Queue of destructor tasks
	std::vector <DeletionTask>	_deletion_tasks;

#ifdef MERCURY_VALIDATION_LAYERS

	// Enabling validation layers
	static constexpr bool enable_validation_layers = true;

#else

	// Disabling validation layers
	static constexpr bool enable_validation_layers = false;

#endif

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(
			window,
			framebuffer_resize_callback
		);
	}

	static void framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast <Vulkan*> (
			glfwGetWindowUserPointer(window)
		);

		// Set the framebuffer resized flag
		app->framebuffer_resized = true;
	}

	void initVulkan() {
		_mk_instance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		_mk_image_views();
		_mk_render_pass();
		_mk_framebuffers();
		_mk_command_pool();
		_mk_command_buffers();
		createSyncObjects();
		_mk_descriptor_set_layout();
		_mk_descriptor_pool();
		_mk_descriptor_sets();
	}

	// TODO: modifiable by the user
	// TODO: method to set default layout, and rebuild descriptor sets
	void _mk_descriptor_set_layout() {
		// Binding info
		VkDescriptorSetLayoutBinding compute_bindings_1 {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr
		};

		VkDescriptorSetLayoutBinding compute_bindings_2 {
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr
		};
		
		VkDescriptorSetLayoutBinding compute_bindings_3 {
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr
		};
		
		VkDescriptorSetLayoutBinding compute_bindings_4 {
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr
		};

		// VkDesciptorSetLayoutBinding
		VkDescriptorSetLayoutBinding compute_bindings[] {
			compute_bindings_1,
			compute_bindings_2,
			compute_bindings_3,
			compute_bindings_4
		};
		
		// Create info
		VkDescriptorSetLayoutCreateInfo layout_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 4,
			.pBindings = &compute_bindings[0]
		};

		// Create the descriptor set layout
		VkResult result = vkCreateDescriptorSetLayout(
			device, &layout_info,
			allocator, &ds_layout
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create descriptor set layout!");
			throw (-1);
		}
	}

	void _mk_descriptor_pool() {
		// Pool sizes
		VkDescriptorPoolSize pool_sizes[] = {
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};

		// Creation info
		VkDescriptorPoolCreateInfo pool_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes),
			.poolSizeCount = (uint32_t) IM_ARRAYSIZE(pool_sizes),
			.pPoolSizes = pool_sizes
		};

		// Creation
		// TODO: wrap inside a method
		VkResult result = vkCreateDescriptorPool(
			device,	&pool_info,
			allocator, &descriptor_pool
		);

		if (result != VK_SUCCESS) {
			throw(-1);
		}
	}

	void _mk_descriptor_sets() {
		// Descriptor set layouts
		descriptor_set_layouts.resize(swch_images.size(), ds_layout);

		// Descriptor set creation info
		VkDescriptorSetAllocateInfo alloc_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = static_cast <uint32_t>
				(descriptor_set_layouts.size()),
			.pSetLayouts = descriptor_set_layouts.data()
		};

		// Creation
		descriptor_sets.resize(swch_images.size());
		VkResult result = vkAllocateDescriptorSets(
			device, &alloc_info, descriptor_sets.data()
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate descriptor sets");
			throw(-1);
		}
	}

	//////////////////////
	// Cleanup routines //
	//////////////////////

	void _cleanup_swapchain() {
		for (auto framebuffer : swch_framebuffers)
			vkDestroyFramebuffer(device, framebuffer, nullptr);

		vkFreeCommandBuffers(
			device, command_pool,
			static_cast <uint32_t> (command_buffers.size()),
			command_buffers.data()
		);

		vkDestroyRenderPass(device, render_pass, nullptr);

		for (auto image_view : swch_image_views)
			vkDestroyImageView(device, image_view, nullptr);

		vkDestroySwapchainKHR(device, swch, nullptr);
	}

	void _cleanup_buffers() {
		for (auto buffer : _buffers) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vkFreeMemory(device, buffer.memory, nullptr);
		}
	}

	void cleanup() {
		// Destroy the swapchain
		_cleanup_swapchain();

		// Destroy descriptor pool
		// vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

		/* Destroy descriptor set layouts
		for (auto layout : descriptor_set_layouts)
			vkDestroyDescriptorSetLayout(device, layout, nullptr); */

		// Destroy descriptor sets
		for (auto set : descriptor_sets)
			vkFreeDescriptorSets(device, descriptor_pool, 1, &set);

		// Destroy synchronization objects
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
			vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
			vkDestroyFence(device, in_flight_fences[i], nullptr);
		}

		vkDestroyCommandPool(device, command_pool, nullptr);

		// Run all deletion tasks
		for (auto &task : _deletion_tasks)
			task(this);

		// _cleanup_buffers();

		vkDestroyDevice(device, nullptr);

		if (enable_validation_layers) {
			_delete_debug_messenger(
				instance, _debug_messenger,
				nullptr
			);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		// Destroy GLFW window
		glfwDestroyWindow(window);

		// End GLFW
		glfwTerminate();
	}

	void _remk_swapchain() {
		// TODO: method to returnn window size
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		// Wait for all frames to finish
		vkDeviceWaitIdle(device);

		// Cleanup swapchain
		_cleanup_swapchain();

		// Recreate swapchain
		createSwapChain();
		_mk_image_views();
		_mk_render_pass();
		_mk_framebuffers();
		_mk_command_buffers();

		// Recreate buffers
		images_in_flight.resize(
			swch_images.size(),
			VK_NULL_HANDLE
		);
	}

	void _mk_instance() {
		if (enable_validation_layers && !_check_validation_layer_support()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enable_validation_layers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
		} else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = _debug_logger;
	}

	void setupDebugMessenger() {
		if (!enable_validation_layers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (_mk_debug_messenger(instance, &createInfo, nullptr, &_debug_messenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physical_device = device;
				break;
			}
		}

		if (physical_device == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = _find_queue_families(physical_device);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {indices.graphics.value(), indices.present.value()};

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enable_validation_layers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physical_device, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphics.value(), 0, &graphics_queue);
		vkGetDeviceQueue(device, indices.present.value(), 0, &present_queue);
	}

	void createSwapChain() {
		SwapchainSupport swchSupport = querySwapChainSupport(physical_device);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swchSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swchSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swchSupport.capabilities);

		uint32_t imageCount = swchSupport.capabilities.minImageCount + 1;
		if (swchSupport.capabilities.maxImageCount > 0 && imageCount > swchSupport.capabilities.maxImageCount) {
			imageCount = swchSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
			| VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		QueueFamilyIndices indices = _find_queue_families(physical_device);
		uint32_t queueFamilyIndices[] = {indices.graphics.value(), indices.present.value()};

		if (indices.graphics != indices.present) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		} else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swchSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swch) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swch, &imageCount, nullptr);
		swch_images.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swch, &imageCount, swch_images.data());

		swch_image_format = surfaceFormat.format;
		swch_extent = extent;
	}

	void _mk_image_views() {
		// Resize first
		swch_image_views.resize(swch_images.size());

		// Fill with new image views
		for (size_t i = 0; i < swch_images.size(); i++) {
			// Creation info
			VkImageViewCreateInfo create_info {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = swch_images[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = swch_image_format,
				.components = {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY
				},
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				}
			};

			// Create image view
			VkResult result = vkCreateImageView(
				device, &create_info,
				nullptr, &swch_image_views[i]
			);

			if (result != VK_SUCCESS) {
				Logger::error("[Vulkan] Failed to create image view!");
				throw (-1);
			}
		}
	}

	void _mk_render_pass() {
		// Create attachment description
		VkAttachmentDescription color_attachment {
			.format = swch_image_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		};

		// Create attachment reference
		VkAttachmentReference color_attachment_ref {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		// Subpasses and dependencies
		VkSubpassDescription subpass {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref
		};

		VkSubpassDependency dependency {
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
				| VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
		};

		// Create render pass
		VkRenderPassCreateInfo render_pass_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &color_attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency
		};

		VkResult result = vkCreateRenderPass(
			device, &render_pass_info,
			nullptr, &render_pass
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create render pass!");
			throw(-1);
		}
	}

	void _mk_framebuffers() {
		swch_framebuffers.resize(swch_image_views.size());

		for (size_t i = 0; i < swch_image_views.size(); i++) {
			// Arrange attachments
			VkImageView attachments[] = {
				swch_image_views[i]
			};

			// Create framebuffer
			VkFramebufferCreateInfo framebuffer_info {
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = render_pass,
				.attachmentCount = 1,
				.pAttachments = attachments,
				.width = swch_extent.width,
				.height = swch_extent.height,
				.layers = 1
			};

			// Safely create framebuffer
			VkResult result = vkCreateFramebuffer(
				device, &framebuffer_info,
				nullptr, &swch_framebuffers[i]
			);

			if (result != VK_SUCCESS) {
				Logger::error("[Vulkan] Failed to create framebuffer!");
				throw(-1);
			}
		}
	}

	void _mk_command_pool() {
		// Find queue family indices
		QueueFamilyIndices indices = _find_queue_families(physical_device);

		// Create command pool
		VkCommandPoolCreateInfo pool_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.queueFamilyIndex = indices.graphics.value()
		};

		VkResult result = vkCreateCommandPool(
			device, &pool_info,
			nullptr, &command_pool
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create command pool!");
			throw(-1);
		}
	}

	// Find memory type for a given type and properties
	uint32_t _find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties mem_props;
		vkGetPhysicalDeviceMemoryProperties(
			physical_device, &mem_props
		);

		for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
			unsigned int props = (mem_props.memoryTypes[i].propertyFlags & properties);
			if ((type_filter & (1 << i)) && props == properties)
				return i;
		}

		Logger::error("[Vulkan] Failed to find suitable memory type!");
		throw(-1);
	}

	void _mk_command_buffers() {
		// Resize command buffers
		command_buffers.resize(swch_framebuffers.size());

		// Command buffer info
		VkCommandBufferAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t) command_buffers.size()
		};


		// Allocate the command buffers
		VkResult result = vkAllocateCommandBuffers(
			device, &alloc_info, command_buffers.data()
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate command buffers!");
			throw(-1);
		}
	}

	void createSyncObjects() {
		image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
		images_in_flight.resize(swch_images.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
					vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
					vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		} else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	SwapchainSupport querySwapChainSupport(VkPhysicalDevice device) {
		SwapchainSupport details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = _find_queue_families(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swchAdequate = false;
		if (extensionsSupported) {
			SwapchainSupport swchSupport = querySwapChainSupport(device);
			swchAdequate = !swchSupport.formats.empty() && !swchSupport.presentModes.empty();
		}

		return indices && extensionsSupported && swchAdequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices _find_queue_families(VkPhysicalDevice device) {
		// Structure to return
		QueueFamilyIndices indices;

		// Get count and fill vector
		uint32_t qfc = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(
			device, &qfc, nullptr
		);

		std::vector <VkQueueFamilyProperties> qfs(qfc);
		vkGetPhysicalDeviceQueueFamilyProperties(
			device, &qfc, qfs.data()
		);

		for (int i = 0; i < qfs.size(); i++) {
			// Check graphics supports
			if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphics = i;

			// Check present support
			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(
				device, i, surface,
				&present_support
			);
			
			if (present_support)
				indices.present = i;

			// Early quit
			if (indices)
				break;
		}

		return indices;
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enable_validation_layers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	////////////////////
	// Shader modules //
	////////////////////
	
	// Read file glob
	static Glob _read_file(const std::string &path) {
		std::ifstream file(path, std::ios::ate | std::ios::binary);

		// Check that the file exists
		if (!file.is_open()) {
			Logger::error() << __PRETTY_FUNCTION__
				<< ": Failed to open file: "
				<< path << std::endl;
			return {};
		}

		// Get the file size
		size_t fsize = file.tellg();
		file.seekg(0);

		// Allocate memory for the file
		Glob buffer(fsize);

		// Read the file
		file.read(buffer.data(), fsize);
		file.close();

		return buffer;
	}

	// Create a shader from glob
	VkShaderModule _mk_shader_module(const Glob &code) {
		// Creation info
		VkShaderModuleCreateInfo create_info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.size(),
			.pCode = reinterpret_cast
				<const uint32_t*> (code.data())
		};

		// Create the shader module
		VkShaderModule shader_module;

		VkResult result = vkCreateShaderModule(
			device, &create_info,
			nullptr, &shader_module
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create shader module");
			return VK_NULL_HANDLE;
		}

		return shader_module;
	}

	//////////////////////////
	// ImGui initialization //
	//////////////////////////
	
	void _init_imgui() {
		// Create descriptor pool
		VkDescriptorPoolSize pool_sizes[] {
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};

		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1000;
		pool_info.poolSizeCount = std::size(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;

		VkDescriptorPool imgui_pool;
		VkResult result = vkCreateDescriptorPool(
			device, &pool_info,
			nullptr, &imgui_pool
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to create descriptor pool");
			return;
		}

		// Initialize the ImGui context
		ImGui::CreateContext();

		// Initialize the ImGui for Vulkan renderer
		ImGui_ImplGlfw_InitForVulkan(window, true);

		//this initializes imgui for Vulkan
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = instance;
		init_info.PhysicalDevice = physical_device;
		init_info.Device = device;
		init_info.Queue = graphics_queue;
		init_info.DescriptorPool = imgui_pool;
		init_info.MinImageCount = 2;
		init_info.ImageCount = 2;
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

		ImGui_ImplVulkan_Init(&init_info, render_pass);

		// Start a new, single use command buffer
		// TODO: make a method for immediate command buffers
		VkCommandBufferAllocateInfo alloc_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkCommandBuffer command_buffer;
		result = vkAllocateCommandBuffers(
			device, &alloc_info, &command_buffer
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to allocate command buffer");
			return;
		}

		// Start recording the command buffer
		VkCommandBufferBeginInfo begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};

		result = vkBeginCommandBuffer(command_buffer, &begin_info);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to begin command buffer");
			return;
		}

		// Create font textures
		ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

		// End recording the command buffer
		result = vkEndCommandBuffer(command_buffer);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to end command buffer");
			return;
		}

		// Submit the command buffer
		VkSubmitInfo submit_info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffer
		};

		result = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to submit command buffer");
			return;
		}

		// Wait for the command buffer to finish
		vkQueueWaitIdle(graphics_queue);

		// Destroy the command buffer
		// TODO: deletion queue function
		vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

		//clear font textures from cpu data
		ImGui_ImplVulkan_DestroyFontUploadObjects();

		/* add the destroy the imgui created structures
		// TODO: also add a deletion queue
		_mainDeletionQueue.push_function([=]() {

				vkDestroyDescriptorPool(_device, imguiPool, nullptr);
				ImGui_ImplVulkan_Shutdown();
				}); */
	}

	/////////////////////////////////////
	// Debugging and validation layers //
	/////////////////////////////////////

	bool _check_validation_layer_support() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static VkResult _mk_debug_messenger(VkInstance instance,
			const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
			const VkAllocationCallbacks* pAllocator,
			VkDebugUtilsMessengerEXT* pDebugMessenger) {
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(
			instance,
			"vkCreateDebugUtilsMessengerEXT"
		);

		if (func != nullptr) {
			return func(
				instance, pCreateInfo,
				pAllocator, pDebugMessenger
			);
		}
		
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	static void _delete_debug_messenger(VkInstance instance,
			VkDebugUtilsMessengerEXT _debug_messenger,
			const VkAllocationCallbacks* pAllocator) {
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(
			instance,
			"vkDestroyDebugUtilsMessengerEXT"
		);

		if (func != nullptr)
			func(instance, _debug_messenger, pAllocator);
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL _debug_logger(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
			void *pUserData) {
		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
			Logger::error() << "[Vulkan Validation Layer] "
				<< pCallbackData->pMessage << std::endl;

#ifdef MERCURY_THROW_ERROR

			throw std::runtime_error("[Vulkan Validation Layer] "
				"An error occured in the validation layer");

#endif

		} else {

#ifndef MERCURY_VALIDATION_ERROR_ONLY

			Logger::notify() << "[Vulkan Validation Layer] "
				<< pCallbackData->pMessage << std::endl;

#endif

		}

		return VK_FALSE;
	}
public:
	// TODO: create a GLFW class and pass it to the constructor
	Vulkan() {
		initWindow();
		initVulkan();
		_init_imgui();
		Logger::ok("[Vulkan] ImGui initialized");
		Logger::ok("[Vulkan] Vulkan instance completely initialized");
	}

	~Vulkan() {
		cleanup();
	}

	// Destructor tasks
	void push_deletion_task(const DeletionTask &task) {
		_deletion_tasks.push_back(task);
	}

	// Render a frame
	void frame();

	// Set command buffer for each frame
	void set_command_buffers(CommandBufferMaker cbm) {
		// Resize command buffers
		command_buffers.resize(swch_framebuffers.size());

		// Command buffer info
		VkCommandBufferAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t) command_buffers.size()
		};

		// Allocate the command buffers
		VkResult result = vkAllocateCommandBuffers(
			device, &alloc_info, command_buffers.data()
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate command buffers!");
			throw(-1);
		}

		for (size_t i = 0; i < command_buffers.size(); i++) {
			// Command buffer creation info
			VkCommandBufferBeginInfo begin_info {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
			};

			// Begin recording
			result = vkBeginCommandBuffer(command_buffers[i], &begin_info);
			if (result != VK_SUCCESS) {
				Logger::error("[Vulkan] Failed to begin"
					" recording command buffer!");
				throw(-1);
			}

			// Command buffer generation
			cbm(this, i);

			// End recording
			result = vkEndCommandBuffer(command_buffers[i]);
			if (result != VK_SUCCESS) {
				Logger::error("[Vulkan] Failed to end"
					" recording command buffer!");
				throw(-1);
			}
		}
	}

	////////////////////////
	// Allocation methods //
	////////////////////////
	
	// Allocate shader
	// TODO: wrap in struct?
	VkShaderModule make_shader(const std::string &path) {
		Glob g = _read_file(path);
		return _mk_shader_module(g);
	}

	// Create a cmmand buffer
	VkCommandBuffer make_command_buffer() {
		VkCommandBufferAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkCommandBuffer command_buffer;

		VkResult result = vkAllocateCommandBuffers(
			device, &alloc_info, &command_buffer
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate command buffer!");
			throw(-1);
		}

		return command_buffer;
	}
	
	// Allocate a buffer
	// TODO: pass buffer propreties as a struct
	void make_buffer(Buffer &bf, size_t size, VkBufferUsageFlags usage) {
		// Buffer creation info
		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE
		};

		VkResult result = vkCreateBuffer(
			device, &buffer_info,
			nullptr, &bf.buffer
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create buffer!");
			// TODO: other exception class
			throw (-1);
		}

		// Allocate memory for the buffer
		VkMemoryRequirements mem_reqs;
		vkGetBufferMemoryRequirements(device, bf.buffer, &mem_reqs);

		VkMemoryAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = mem_reqs.size,
			.memoryTypeIndex = _find_memory_type(
				mem_reqs.memoryTypeBits,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
			)
		};

		// Create device memory
		result = vkAllocateMemory(
			device, &alloc_info,
			nullptr, &bf.memory
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate buffer memory!");
			throw (-1);
		}

		// Bind the buffer to the memory
		vkBindBufferMemory(device, bf.buffer, bf.memory, 0);

		// Set the buffer properties
		bf.size = size;
		bf.offset = 0;

		// Log creation
		Logger::ok() << "[Vulkan] Buffer created (VkBuffer="
			<< bf.buffer << ", VkDeviceMemory=" << bf.memory
			<< ", size=" << size << ")\n";
	}

	// Destroy a buffer
	void destroy_buffer(Buffer &bf) {
		vkDestroyBuffer(device, bf.buffer, nullptr);
		vkFreeMemory(device, bf.memory, nullptr);
	}

	// Map data to a buffer
	void map_buffer(Buffer *buffer, void *data, size_t size) {
		// Map the buffer
		VkResult result = vkMapMemory(
			device, buffer->memory, 0,
			size, 0, &buffer->mapped
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to map buffer memory!");
			throw (-1);
		}

		// Copy the data
		memcpy(buffer->mapped, data, size);

		// Unmap the buffer
		vkUnmapMemory(device, buffer->memory);
	}
	
	// Create a render pass
	VkRenderPass make_render_pass(VkImageLayout initial_layout, VkImageLayout final_layout) {
		// Render pass to return
		VkRenderPass new_render_pass = VK_NULL_HANDLE;

		// Create attachment description
		VkAttachmentDescription color_attachment {
			.format = swch_image_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = initial_layout,
			.finalLayout = final_layout
		};

		// Create attachment reference
		VkAttachmentReference color_attachment_ref {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		// Subpasses and dependencies
		VkSubpassDescription subpass {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref
		};

		VkSubpassDependency dependency {
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
				| VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
		};

		// Create render pass
		VkRenderPassCreateInfo render_pass_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &color_attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency
		};

		VkResult result = vkCreateRenderPass(
			device, &render_pass_info,
			nullptr, &new_render_pass
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create render pass!");
			throw(-1);
		}

		// Log creation
		Logger::ok() << "[Vulkan] Render pass created (VkRenderPass="
			<< render_pass << ")\n";

		return new_render_pass;
	}

	// Getters
	VkPhysicalDeviceProperties phdev_props() const;

	// Other methods
	void idle() const {
		vkDeviceWaitIdle(device);
	}
};

#endif
