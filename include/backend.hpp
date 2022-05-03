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
#include <vector>

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
#include "logger.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

namespace kobra {

// TODO: aux class which stores device and physcial device
// (and other per device objects)
class Vulkan {
public:
	////////////////////
	// Public aliases //
	////////////////////

	using DS = VkDescriptorSet;
	using DSL = VkDescriptorSetLayout;
	using DSLB = VkDescriptorSetLayoutBinding;

	using VB = VkVertexInputBindingDescription;
	using VA = VkVertexInputAttributeDescription;

	///////////////////////
	// Public structures //
	///////////////////////

	// Buffer structure
	struct Buffer {
		VkBuffer	buffer = VK_NULL_HANDLE;
		VkDeviceMemory	memory = VK_NULL_HANDLE;
		VkDeviceSize	size;
		VkDeviceSize	offset;
		void *		mapped;

		// No destructor, because copying
		// and assignment would be annoying
	};

	// Surface structure
	struct Surface {
		GLFWwindow *	window;
		VkSurfaceKHR	surface;
	};

	// Logical device structure
	struct Device {
		VkDevice	device;
		VkQueue		graphics_queue;
		VkQueue		present_queue;
	};

	// Swapchain structure
	struct Swapchain {
		// Variables
		VkSwapchainKHR			swch;
		VkExtent2D			extent;
		VkFormat			image_format;

		std::vector <VkFramebuffer>	framebuffers;
		std::vector <VkImage>		images;
		std::vector <VkImageView>	image_views;
	};

	// Pipeline structure
	struct Pipeline {
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;
	};

	// Pipeline creation structure
	struct PipelineInfo {
		Swapchain		swapchain;

		VkRenderPass		render_pass;

		VkShaderModule		vert;
		VkShaderModule		frag;

		std::vector <DSL>	dsls;

		VB			vertex_binding;
		std::vector <VA>	vertex_attributes;

		size_t			push_consts;
		VkPushConstantRange	*push_consts_range;

		bool			depth_test;

		VkPrimitiveTopology	topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		struct {
			// TODO: make floats
			int				width = 0;
			int				height = 0;
			int				x = 0;
			int				y = 0;
		} viewport;
	};

	// Context, containing the device
	// TODO: include swapchain?
	// TODO: put outside of this class?
	struct Context {
		Vulkan			*vk;
		Device			device;
		VkPhysicalDevice	phdev;

		// Get underlying device
		const VkDevice &vk_device() const {
			return device.device;
		}

		// Create a shader module
		VkShaderModule make_shader(const std::string &path) const {
			Glob g = vk->_read_file(path);
			return vk->_mk_shader_module(device, g);
		}

		// Create multiple shader modules
		std::vector <VkShaderModule> make_shaders(const std::vector <std::string> &paths) const {
			std::vector <VkShaderModule> modules;
			modules.reserve(paths.size());
			for (const auto &path : paths)
				modules.push_back(make_shader(path));
			return modules;
		}

		// Create descritpor set layout
		DSL make_dsl(const std::vector <DSLB> &bindings) const {
			return vk->make_descriptor_set_layout(device, bindings);
		}

		// Create a descriptor set
		DS make_ds(const VkDescriptorPool &pool, const DSL &layout) const {
			return vk->make_descriptor_set(device, pool, layout);
		}

		// Get supported formats
		VkFormat find_supported_format(const std::vector <VkFormat> &,
				const VkImageTiling &,
				const VkFormatFeatureFlags &) const;

		// Get supported depth format
		VkFormat find_depth_format() const;

		// Create a render pass
		VkRenderPass make_render_pass(const Swapchain &swapchain,
				const VkAttachmentLoadOp &load_op,
				const VkAttachmentStoreOp &store_op) {
			return vk->make_render_pass(phdev, device,
				swapchain, load_op, store_op
			);
		}

		// Create a graphics pipeline
		Pipeline make_pipeline(const PipelineInfo &info) const {
			// Create pipeline stages
			VkPipelineShaderStageCreateInfo vertex {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = info.vert,
				.pName = "main"
			};

			VkPipelineShaderStageCreateInfo fragment {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = info.frag,
				.pName = "main"
			};

			VkPipelineShaderStageCreateInfo shader_stages[] = { vertex, fragment };

			// Create vertex input state
			VkPipelineVertexInputStateCreateInfo vertex_input {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &info.vertex_binding,
				.vertexAttributeDescriptionCount
					= static_cast <uint32_t> (info.vertex_attributes.size()),
				.pVertexAttributeDescriptions = info.vertex_attributes.data()
			};

			// Input assembly
			VkPipelineInputAssemblyStateCreateInfo input_assembly {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = info.topology,
				.primitiveRestartEnable = VK_FALSE
			};

			// Viewport
			VkViewport viewport {
				.x = (float) info.viewport.x,
				.y = (float) info.viewport.y,
				.width = (float) info.viewport.width,
				.height = (float) info.viewport.height,
				.minDepth = 0.0f,
				.maxDepth = 1.0f
			};

			// Scissor
			VkRect2D scissor {
				.offset = {0, 0},
				.extent = info.swapchain.extent
			};

			VkPipelineViewportStateCreateInfo viewport_state {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.pViewports = &viewport,
				.scissorCount = 1,
				.pScissors = &scissor
			};

			// Rasterizer
			// TODO: method
			VkPipelineRasterizationStateCreateInfo rasterizer {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.depthClampEnable = VK_FALSE,
				.rasterizerDiscardEnable = VK_FALSE,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_NONE,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.depthBiasEnable = VK_FALSE,
				.lineWidth = 1.0f
			};

			// Multisampling
			// TODO: method
			VkPipelineMultisampleStateCreateInfo multisampling {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
				.sampleShadingEnable = VK_FALSE,
				.minSampleShading = 1.0f,
				.pSampleMask = nullptr,
				.alphaToCoverageEnable = VK_FALSE,
				.alphaToOneEnable = VK_FALSE
			};

			// Color blending
			VkPipelineColorBlendAttachmentState color_blend_attachment {
				.blendEnable = VK_TRUE,
				.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
				.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
				.colorBlendOp = VK_BLEND_OP_ADD,
				.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
				.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA,
				.alphaBlendOp = VK_BLEND_OP_MAX,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
					| VK_COLOR_COMPONENT_G_BIT
					| VK_COLOR_COMPONENT_B_BIT
					| VK_COLOR_COMPONENT_A_BIT
			};

			VkPipelineColorBlendStateCreateInfo color_blending {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.logicOpEnable = VK_FALSE,
				.logicOp = VK_LOGIC_OP_COPY,
				.attachmentCount = 1,
				.pAttachments = &color_blend_attachment,
				.blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
			};

			// Pipeline layout
			VkPipelineLayoutCreateInfo pipeline_layout_info {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = static_cast <uint32_t> (info.dsls.size()),
				.pSetLayouts = info.dsls.data(),
				.pushConstantRangeCount = static_cast <uint32_t> (info.push_consts),
				.pPushConstantRanges = info.push_consts_range
			};

			VkPipelineLayout pipeline_layout;
			VkResult result = vkCreatePipelineLayout(
				vk_device(),
				&pipeline_layout_info,
				nullptr,
				&pipeline_layout
			);

			if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to create pipeline layout!");
			}

			// Depth stencil if requested
			VkPipelineDepthStencilStateCreateInfo depth_stencil {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = info.depth_test,
				.depthWriteEnable = info.depth_test, // TODO: depth test struct
				.depthCompareOp = VK_COMPARE_OP_LESS,
				.depthBoundsTestEnable = VK_FALSE,
				.stencilTestEnable = VK_FALSE,
				.front = {},
				.back = {},
				.minDepthBounds = 0.0f,
				.maxDepthBounds = 1.0f
			};

			// Graphics pipeline
			VkGraphicsPipelineCreateInfo pipeline_info {
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = shader_stages,
				.pVertexInputState = &vertex_input,
				.pInputAssemblyState = &input_assembly,
				.pViewportState = &viewport_state,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pDepthStencilState = &depth_stencil,
				.pColorBlendState = &color_blending,
				.pDynamicState = nullptr,
				.layout = pipeline_layout,
				.renderPass = info.render_pass,
				.subpass = 0,
				.basePipelineHandle = VK_NULL_HANDLE,
				.basePipelineIndex = -1
			};

			VkPipeline pipeline;
			result = vkCreateGraphicsPipelines(
				vk_device(),
				VK_NULL_HANDLE,
				1,
				&pipeline_info,
				nullptr,
				&pipeline
			);

			if (result != VK_SUCCESS) {
				KOBRA_LOG_FUNC(error) << "Failed to create graphics pipeline!\n";
				return {VK_NULL_HANDLE, VK_NULL_HANDLE};
			}

			return {pipeline, pipeline_layout};
		}
	};

	// Aliases
	using Glob = std::vector <char>;

	// TODO: depreciate this version
	using CommandBufferMaker = std::function <void (const Vulkan *, size_t)>;
	using DeletionTask = std::function <void (Vulkan *)>;	// TODO: Is this Vulkan object needed?

	using DSLayout = VkDescriptorSetLayout;

	// Vulkan basic context
	VkInstance instance;
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
		std::vector <VkPresentModeKHR>		present_modes;
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

#ifdef KOBRA_VALIDATION_LAYERS

	// Enabling validation layers
	static constexpr bool enable_validation_layers = true;

#else

	// Disabling validation layers
	static constexpr bool enable_validation_layers = false;

#endif

	static void framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast <Vulkan*> (
			glfwGetWindowUserPointer(window)
		);

		// Set the framebuffer resized flag
		app->framebuffer_resized = true;
	}

	void _init_vulkan() {
		_mk_instance();
		_setup_debug_messenger();
	}

	// TODO: modifiable by the user
	// TODO: method to set default layout, and rebuild descriptor sets

	//////////////////////
	// Cleanup routines //
	//////////////////////

	void cleanup() {
		// Run all deletion tasks
		for (auto &task : _deletion_tasks)
			task(this);

		if (enable_validation_layers) {
			_delete_debug_messenger(
				instance, _debug_messenger,
				nullptr
			);
		}

		// vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		// End GLFW
		glfwTerminate();
	}

	void _mk_instance() {
		if (enable_validation_layers && !_check_validation_layer_support()) {
			Logger::error("[Vulkan] Validation layers requested, but not available");
			throw -1;
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Mercury";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "Mercury";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &appInfo;

		auto extensions = _get_required_extensions();
		create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		create_info.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
		if (enable_validation_layers) {
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();

			_populate_debug_messenger_create_info(debug_create_info);
			create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debug_create_info;
		} else {
			create_info.enabledLayerCount = 0;
			create_info.pNext = nullptr;
		}

		if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create instance");
			throw -1;
		}
	}

	void _populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
		create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = _debug_logger;
	}

	void _setup_debug_messenger() {
		if (!enable_validation_layers) return;

		VkDebugUtilsMessengerCreateInfoEXT create_info;
		_populate_debug_messenger_create_info(create_info);

		if (_mk_debug_messenger(instance, &create_info, nullptr, &_debug_messenger) != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to set up debug messenger");
			throw -1;
		}
	}

	void _make_image_views(const Device &, Swapchain &) const;

	// TODO: depreciate

	// Find memory type for a given type and properties
	// TODO: public overload only
	uint32_t _find_memory_type(const VkPhysicalDevice &phdev,
			uint32_t type_filter,
			VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties mem_props;
		vkGetPhysicalDeviceMemoryProperties(
			phdev, &mem_props
		);

		for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
			unsigned int props = (mem_props.memoryTypes[i].propertyFlags & properties);
			if ((type_filter & (1 << i)) && props == properties)
				return i;
		}

		Logger::error("[Vulkan] Failed to find suitable memory type!");
		throw(-1);
	}

	VkSurfaceFormatKHR _choose_swch_surface_format(const std::vector <VkSurfaceFormatKHR> &fmts) {
		for (const auto &fmt : fmts) {
			if (fmt.format == VK_FORMAT_R8G8B8A8_SINT
					&& fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return fmt;
		}

		return fmts[0];
	}

	VkPresentModeKHR _choose_swch_present_mode(const std::vector <VkPresentModeKHR> &pmodes) {
		return VK_PRESENT_MODE_IMMEDIATE_KHR;
	}

	VkExtent2D _choose_swch_extent(const Surface &surface, const VkSurfaceCapabilitiesKHR &capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX)
			return capabilities.currentExtent;

		int width, height;
		glfwGetFramebufferSize(surface.window, &width, &height);

		VkExtent2D ext = {
			static_cast <uint32_t> (width),
			static_cast <uint32_t> (height)
		};

		ext.width = std::clamp(ext.width,
			capabilities.minImageExtent.width,
			capabilities.maxImageExtent.width
		);

		ext.height = std::clamp(ext.height,
			capabilities.minImageExtent.height,
			capabilities.maxImageExtent.height
		);

		return ext;
	}

	SwapchainSupport _query_swch_support(const VkPhysicalDevice &device, const Surface &surface) const {
		SwapchainSupport details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			device, surface.surface,
			&details.capabilities
		);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(
			device, surface.surface,
			&format_count, nullptr
		);

		if (format_count != 0) {
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(
				device, surface.surface,
				&format_count,
				details.formats.data()
			);
		}

		uint32_t present_mode_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(
			device, surface.surface,
			&present_mode_count, nullptr
		);

		if (present_mode_count != 0) {
			details.present_modes.resize(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(
				device, surface.surface,
				&present_mode_count,
				details.present_modes.data()
			);
		}

		return details;
	}

	bool _check_device_suitability(const VkPhysicalDevice &device, const Surface &surface) const {
		QueueFamilyIndices indices = _find_queue_families(device, surface);

		bool extensions_supported = checkDeviceExtensionSupport(device);
		bool swch_adequate = false;

		if (extensions_supported) {
			SwapchainSupport swch_support = _query_swch_support(device, surface);
			swch_adequate = !swch_support.formats.empty()
				&& !swch_support.present_modes.empty();
		}

		return indices && extensions_supported && swch_adequate;
	}

	// TODO: refactor
	bool checkDeviceExtensionSupport(const VkPhysicalDevice &device) const {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(device_extensions.begin(), device_extensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices _find_queue_families(const VkPhysicalDevice &device, const Surface &surface) const {
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
				device, i, surface.surface,
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

	std::vector <const char *> _get_required_extensions() {
		uint32_t glfw_ext_count = 0;
		const char **glfw_exts;
		glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

		std::vector <const char *> extensions(glfw_exts, glfw_exts + glfw_ext_count);
		extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
		extensions.push_back("VK_KHR_get_physical_device_properties2");

		if (enable_validation_layers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		return extensions;
	}

	// Get supported formats
	VkFormat _find_supported_format(const VkPhysicalDevice &,
			const std::vector <VkFormat> &,
			const VkImageTiling &,
			const VkFormatFeatureFlags &) const;

	// Get supported depth format
	VkFormat _find_depth_format(const VkPhysicalDevice &) const;

	////////////////////
	// Shader modules //
	////////////////////

	// Read file glob
	static Glob _read_file(const std::string &path) {
		std::ifstream file(path, std::ios::ate | std::ios::binary);

		// Check that the file exists
		if (!file.is_open()) {
			KOBRA_LOG_FUNC(error) << "Failed to open file: " << path << std::endl;
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
	VkShaderModule _mk_shader_module(const Device &device, const Glob &code) {
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
			device.device, &create_info,
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

	// TODO: public method

	// TODO: separate from this backend class

	/////////////////////////////////////
	// Debugging and validation layers //
	/////////////////////////////////////

	bool _check_validation_layer_support() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validation_layers) {
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
		// Errors
		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
			Logger::error() << "[Vulkan Validation Layer] "
				<< pCallbackData->pMessage << std::endl;

#ifdef KOBRA_THROW_ERROR

			throw std::runtime_error("[Vulkan Validation Layer] "
				"An error occured in the validation layer");

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
public:
	// TODO: create a GLFW class and pass it to the constructor
	Vulkan() {
		// _init_window();
		// TODO: initglfw function
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		_init_vulkan();
		Logger::ok("[Vulkan] Vulkan instance completely initialized");
	}

	~Vulkan() {
		cleanup();
	}

	// TODO: move to global scope

	// ImGui window context variables
	struct ImGuiContext {
		VkDescriptorPool	descriptor_pool;
		VkCommandPool		command_pool;
		VkCommandBuffer		command_buffer;
		VkRenderPass		render_pass;
		VkSemaphore		semaphore;
		VkFence			fence;
	};

	// Destructor tasks
	void push_deletion_task(const DeletionTask &task) {
		_deletion_tasks.push_back(task);
	}

	// TODO: pop deletion tasks

	// Create an image
	void make_image(const VkPhysicalDevice &phdev, const VkDevice &device,
			uint32_t width, uint32_t height,
			VkFormat format, VkImageTiling tiling,
			VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
			VkImage &image, VkDeviceMemory &imageMemory) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = find_memory_type(phdev, memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	// Create an image view
	VkImageView make_image_view(const VkDevice &device,
			VkImage image, VkFormat format,
			VkImageAspectFlags aspectFlags) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	// Single use command buffers
	static VkCommandBuffer begin_single_time_commands(const Context &ctx, const VkCommandPool &pool)
	{
		// Create command buffer
		VkCommandBufferAllocateInfo alloc_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkCommandBuffer tmp_cmd_buffer;
		vkAllocateCommandBuffers(ctx.device.device, &alloc_info, &tmp_cmd_buffer);

		// Start recording the command buffer
		VkCommandBufferBeginInfo begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};

		vkBeginCommandBuffer(tmp_cmd_buffer, &begin_info);

		return tmp_cmd_buffer;
	}

	static void submit_single_time_commands(const Context &ctx, const VkCommandPool &pool, const VkCommandBuffer &cmd_buffer)
	{
		// End recording the command buffer
		vkEndCommandBuffer(cmd_buffer);

		// Submit the command buffer
		VkSubmitInfo submit_info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd_buffer
		};

		vkQueueSubmit(ctx.device.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);

		// Wait for the command buffer to finish
		vkQueueWaitIdle(ctx.device.graphics_queue);

		// Destroy the command buffer
		vkFreeCommandBuffers(ctx.vk_device(), pool, 1, &cmd_buffer);
	}

	// Set command buffer for each frame
	void set_command_buffers(const Device &device,
			const Swapchain &swch,
			VkCommandPool cpool,
			std::vector <VkCommandBuffer> &buffers,
			CommandBufferMaker maker) const {
		VkResult result;

		// Free old command buffers
		for (auto &buffer : buffers)
			vkFreeCommandBuffers(device.device, cpool, 1, &buffer);

		// Allocate new command buffers
		buffers.resize(swch.framebuffers.size());

		// Command buffer info
		VkCommandBufferAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = cpool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t) buffers.size()
		};

		// Allocate the command buffers
		result = vkAllocateCommandBuffers(
			device.device, &alloc_info,
			buffers.data()
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate command buffers!");
			throw(-1);
		}

		for (size_t i = 0; i < buffers.size(); i++) {
			// Command buffer creation info
			VkCommandBufferBeginInfo begin_info {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
			};

			// Begin recording
			result = vkBeginCommandBuffer(buffers[i], &begin_info);
			if (result != VK_SUCCESS) {
				Logger::error("[Vulkan] Failed to begin"
					" recording command buffer!");
				throw(-1);
			}

			// Command buffer generation
			maker(this, i);

			// End recording
			result = vkEndCommandBuffer(buffers[i]);
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

	uint32_t find_memory_type(const VkPhysicalDevice &phdev,
			uint32_t type_filter,
			VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties mem_props;
		vkGetPhysicalDeviceMemoryProperties(
			phdev, &mem_props
		);

		for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
			unsigned int props = (mem_props.memoryTypes[i].propertyFlags & properties);
			if ((type_filter & (1 << i)) && props == properties)
				return i;
		}

		Logger::error("[Vulkan] Failed to find suitable memory type!");
		throw(-1);
	}

	// Allocate shader
	// TODO: wrap in struct?
	VkShaderModule make_shader(const Device &device, const std::string &path) {
		Glob g = _read_file(path);
		return _mk_shader_module(device, g);
	}

	// Create a command buffer
	// TODO: pass level
	VkCommandBuffer make_command_buffer(const Device &device, VkCommandPool cmd_pool) {
		VkCommandBufferAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = cmd_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkCommandBuffer command_buffer;

		VkResult result = vkAllocateCommandBuffers(
			device.device, &alloc_info, &command_buffer
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate command buffer!");
			throw(-1);
		}

		return command_buffer;
	}

	// Creating multiple command buffers
	void make_command_buffers(const Device &device, VkCommandPool command_pool, std::vector <VkCommandBuffer> &buffers, size_t size) const {
		// First resize
		buffers.resize(size);

		// Fill command buffer info
		VkCommandBufferAllocateInfo alloc_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = static_cast <uint32_t> (size)
		};

		// Allocate the command buffers
		VkCommandBuffer command_buffer;
		VkResult result = vkAllocateCommandBuffers(
			device.device, &alloc_info, buffers.data()
		);

		// Check for errors
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate command buffer(s)!");
			throw(-1);
		}
	}

	// Start and end recording a command buffer
	void begin_command_buffer(VkCommandBuffer command_buffer) {
		VkCommandBufferBeginInfo begin_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
		};

		VkResult result = vkBeginCommandBuffer(command_buffer, &begin_info);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to begin"
				" recording command buffer!");
			throw(-1);
		}
	}

	void end_command_buffer(VkCommandBuffer command_buffer) {
		VkResult result = vkEndCommandBuffer(command_buffer);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to end"
				" recording command buffer!");
			throw(-1);
		}
	}

	// Submit a command buffer
	void submit_command_buffer(const Device &device, VkCommandBuffer command_buffer) {
		VkSubmitInfo submit_info {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffer
		};

		VkResult result = vkQueueSubmit(
			device.graphics_queue, 1, &submit_info, VK_NULL_HANDLE
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to submit command buffer!");
			throw(-1);
		}
	}

	// Buffer methods
	// TODO: pass buffer propreties as a struct
	void make_buffer(const VkPhysicalDevice &, const Device &, Buffer &, size_t, VkBufferUsageFlags);
	void destroy_buffer(const Device &, Buffer &);

	void map_buffer(const Device &, Buffer *, void *, size_t);

	void *get_buffer_data(const Device &device, const Buffer &buffer) {
		void *data;
		vkMapMemory(device.device, buffer.memory, 0, buffer.size, 0, &data);
		return data;
	}

	// Create a render pass
	VkRenderPass make_render_pass(const VkPhysicalDevice &phdev,
			const Device &device,
			const Swapchain &swch,
			VkAttachmentLoadOp,
			VkAttachmentStoreOp,
			VkImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			VkImageLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) const;

	// Start and end a render pass
	void begin_render_pass(VkCommandBuffer, VkFramebuffer,
			VkRenderPass, VkExtent2D,
			uint32_t, VkClearValue *) const;
	void end_render_pass(VkCommandBuffer cmd_buffer) const;

	// Create a command pool
	VkCommandPool make_command_pool(const VkPhysicalDevice &phdev,
			const Surface &surface,
			const Device &device,
			const VkCommandPoolCreateFlags flags) {
		// Command pool to return
		VkCommandPool new_command_pool = VK_NULL_HANDLE;

		// Find queue family indices
		QueueFamilyIndices indices = _find_queue_families(phdev, surface);

		// Create command pool
		VkCommandPoolCreateInfo pool_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = flags,
			.queueFamilyIndex = indices.graphics.value()
		};

		VkResult result = vkCreateCommandPool(
			device.device, &pool_info,
			nullptr, &new_command_pool
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create command pool!");
			throw(-1);
		}

		return new_command_pool;
	}

	// Create a semaphore
	VkSemaphore make_semaphore(const Device &device) {
		// Semaphore
		VkSemaphore new_semaphore = VK_NULL_HANDLE;

		// Create semaphore
		VkSemaphoreCreateInfo semaphore_info {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};

		VkResult result = vkCreateSemaphore(
			device.device, &semaphore_info,
			nullptr, &new_semaphore
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create semaphore!");
			throw(-1);
		}

		return new_semaphore;
	}

	// Create a fence
	VkFence make_fence(const Device &device, VkFenceCreateFlags flags = 0) {
		// Fence
		VkFence new_fence = VK_NULL_HANDLE;

		// Create fence
		VkFenceCreateInfo fence_info {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = flags
		};

		VkResult result = vkCreateFence(
			device.device, &fence_info,
			nullptr, &new_fence
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create fence!");
			throw(-1);
		}

		return new_fence;
	}

	// Create a window surface
	Surface make_surface(const std::string &title, int width, int height) {
		// Create the GLFW window
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		GLFWwindow *window = glfwCreateWindow(
			width, height, title.c_str(),
			nullptr, nullptr
		);

		// Create the surface
		VkSurfaceKHR new_surface = VK_NULL_HANDLE;
		VkResult result = glfwCreateWindowSurface(
			instance, window,
			nullptr, &new_surface
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create surface!");
			throw(-1);
		}

		// Log creation
		return Surface {window, new_surface};
	}

	// Create a swapchain and related functions
	Swapchain make_swapchain(const VkPhysicalDevice &, const Device &device, const Surface &);
	void make_framebuffers(const Device &, Swapchain &, VkRenderPass,
			const std::vector <VkImageView> & = {}) const;

	// Create a descriptor pool
	// TODO: pass sizes (and a default) in a struct
	VkDescriptorPool make_descriptor_pool(const Device &device, VkAllocationCallbacks *allocator = nullptr) const {
		// Descriptor pool to return
		VkDescriptorPool new_descriptor_pool = VK_NULL_HANDLE;

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
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
				| VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
			.maxSets = 1000 * sizeof(pool_sizes) / sizeof(VkDescriptorPoolSize),
			.poolSizeCount = (uint32_t) sizeof(pool_sizes) / sizeof(VkDescriptorPoolSize),
			.pPoolSizes = pool_sizes
		};

		// Creation
		// TODO: wrap inside a method
		VkResult result = vkCreateDescriptorPool(
			device.device, &pool_info,
			allocator, &new_descriptor_pool
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create descriptor pool!");
			throw(-1);
		}

		// Log creation and return
		return new_descriptor_pool;
	}

	// Create a descriptor set layout
	VkDescriptorSetLayout make_descriptor_set_layout(const Device &device,
			const std::vector <VkDescriptorSetLayoutBinding> &bindings,
			const VkDescriptorSetLayoutCreateFlags &flags = 0,
			VkAllocationCallbacks *allocator = nullptr) const {
		// Descriptor set layout to return
		VkDescriptorSetLayout new_descriptor_set_layout = VK_NULL_HANDLE;

		// Create info
		VkDescriptorSetLayoutCreateInfo layout_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.flags = flags,
			.bindingCount = (uint32_t) bindings.size(),
			.pBindings = bindings.data()
		};

		// Create the descriptor set layout
		VkResult result = vkCreateDescriptorSetLayout(
			device.device, &layout_info,
			allocator, &new_descriptor_set_layout
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create descriptor set layout!");
			throw (-1);
		}

		// Log creation and return
		return new_descriptor_set_layout;
	}

	// Create a descriptor set
	VkDescriptorSet make_descriptor_set(const Device &device, VkDescriptorPool dpool, VkDescriptorSetLayout dsl) const {
		// Descriptor set to return
		VkDescriptorSet new_descriptor_set = VK_NULL_HANDLE;

		// Descriptor set creation info
		VkDescriptorSetAllocateInfo alloc_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = dpool,
			.descriptorSetCount = 1,
			.pSetLayouts = &dsl
		};

		// Creation
		VkResult result = vkAllocateDescriptorSets(
			device.device, &alloc_info,
			&new_descriptor_set
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to allocate descriptor sets");
			throw(-1);
		}

#ifdef KOBRA_LOG_ALL

		Logger::ok() << "[Vulkan] Descriptor set created (VkDescriptorSet="
			<< new_descriptor_set << ")\n";

#endif

		return new_descriptor_set;
	}

	// Device creation
	VkPhysicalDevice select_phdev(const Surface &surface) const {
		// Physical device to return
		VkPhysicalDevice new_physical_device = VK_NULL_HANDLE;

		// Get the number of physical devices
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(
			instance,
			&device_count,
			nullptr
		);

		if (device_count == 0) {
			Logger::error("[Vulkan] No physical devices found!");
			throw(-1);
		}

		// Get the physical devices
		std::vector <VkPhysicalDevice> devices(device_count);
		vkEnumeratePhysicalDevices(
			instance,
			&device_count,
			devices.data()
		);

		for (const auto &device : devices) {
			if (_check_device_suitability(device, surface)) {
				new_physical_device = device;
				break;
			}
		}

		if (new_physical_device == VK_NULL_HANDLE)
			Logger::warn("[Vulkan] Failed to find a suitable GPU!");

		// Log creation and return
		Logger::ok() << "[Vulkan] Physical device selected (VkPhysicalDevice="
			<< new_physical_device << ")\n";
		return new_physical_device;
	}

	Device make_device(const VkPhysicalDevice &phdev, const Surface &surface) const {
		// Device to return
		Device new_device {
			.device = VK_NULL_HANDLE,
			.graphics_queue = VK_NULL_HANDLE,
			.present_queue = VK_NULL_HANDLE
		};

		// Queue family indices
		QueueFamilyIndices indices = _find_queue_families(phdev, surface);

		std::vector <VkDeviceQueueCreateInfo> queue_create_infos;
		std::set <uint32_t> unique_queue_families = {
			indices.graphics.value(),
			indices.present.value()
		};

		float queue_priority = 1.0f;
		for (uint32_t queue_family : unique_queue_families) {
			VkDeviceQueueCreateInfo queue_create_info {
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queue_family,
				.queueCount = 1,
				.pQueuePriorities = &queue_priority
			};

			queue_create_infos.push_back(queue_create_info);
		}

		VkPhysicalDeviceFeatures device_features {};
		VkDeviceCreateInfo create_info {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = (uint32_t) queue_create_infos.size(),
			.pQueueCreateInfos = queue_create_infos.data(),
			.enabledExtensionCount = (uint32_t) device_extensions.size(),
			.ppEnabledExtensionNames = device_extensions.data(),
			.pEnabledFeatures = &device_features
		};

		if (enable_validation_layers) {
			create_info.enabledLayerCount = static_cast <uint32_t> (validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		} else {
			create_info.enabledLayerCount = 0;
		}

		if (vkCreateDevice(phdev, &create_info, nullptr, &new_device.device) != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create logical device!");
			throw(-1);
		}

		// Create the device queues
		vkGetDeviceQueue(
			new_device.device,
			indices.graphics.value(),
			0, &new_device.graphics_queue
		);

		vkGetDeviceQueue(
			new_device.device,
			indices.present.value(),
			0, &new_device.present_queue
		);

		// Log creation and return
		Logger::ok() << "[Vulkan] Device created (VkDevice="
			<< new_device.device << ")\n";
		return new_device;
	}

	// Getters
	VkPhysicalDeviceProperties phdev_props(const VkPhysicalDevice &) const;

	// Other methods
	void idle(const Device &) const;

	// Static methods
	static void begin(const VkCommandBuffer cbuf) {
		VkCommandBufferBeginInfo begin_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
		};

		vkBeginCommandBuffer(cbuf, &begin_info);
	}

	static void end(const VkCommandBuffer cbuf) {
		vkEndCommandBuffer(cbuf);
	}

	static void begin_render_pass(const VkCommandBuffer &cmd,
			const VkRenderPass &render_pass,
			const VkFramebuffer &framebuffer,
			const VkRect2D &render_area,
			const std::vector <VkClearValue> &clear_values) {
		VkRenderPassBeginInfo render_pass_info = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea = render_area,
			.clearValueCount = (uint32_t) clear_values.size(),
			.pClearValues = clear_values.data()
		};

		vkCmdBeginRenderPass(cmd,
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);
	}

	static void end_render_pass(const VkCommandBuffer &cmd) {
		vkCmdEndRenderPass(cmd);
	}

	// Static member variables
	static const std::vector <const char *> device_extensions;
	static const std::vector <const char *> validation_layers;
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

// Get (or create) the singleton instance
inline const vk::raii::Instance &get_vulkan_instance()
{
	static vk::ApplicationInfo app_info {
		"Kobra",
		VK_MAKE_VERSION(1, 0, 0),
		"Kobra",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_0
	};

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

	static vk::raii::Instance instance {
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

	return instance;
}


// Initialize GLFW statically
inline void _initialize_glfw()
{
	static bool initialized = false;

	if (!initialized) {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		initialized = true;
	}
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
};

// Create a surface given a window
inline vk::raii::SurfaceKHR make_surface(const Window &window)
{
	// Create the surface
	VkSurfaceKHR surface;
	glfwCreateWindowSurface(
		*get_vulkan_instance(),
		window.handle,
		nullptr, &surface
	);

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

// Create a logical device
inline vk::raii::Device make_device(const vk::raii::PhysicalDevice &phdev,
		const QueueFamilyIndices &indices,
		const std::vector <const char *> &extensions)
{
	float queue_priority = 0.0f;

	// Create the device info
	vk::DeviceQueueCreateInfo queue_info {
		vk::DeviceQueueCreateFlags(),
		indices.graphics, 1, &queue_priority
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
	default:
		KOBRA_ASSERT(false, "Unsupported old layout");
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
	default:
		KOBRA_ASSERT(false, "Unsupported old layout");
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
		KOBRA_ASSERT(false, "Unsupported new layout");
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
		KOBRA_ASSERT(false, "Unsupported new layout");
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
	vk::Format format;
	vk::raii::Image		image = nullptr;
	vk::raii::DeviceMemory	memory = nullptr;
	vk::raii::ImageView	view = nullptr;

	// Constructors
	ImageData(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::Format &fmt,
			const vk::Extent2D &extent,
			vk::ImageTiling tiling,
			vk::ImageUsageFlags usage,
			vk::ImageLayout initial_layout,
			vk::MemoryPropertyFlags memory_properties,
			vk::ImageAspectFlags aspect_mask)
			: format { fmt },

			image { device,
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
					  initial_layout
				  }
			},

			memory {
				allocate_device_memory(
					device, phdev.getMemoryProperties(),
					image.getMemoryRequirements(),
					memory_properties
				)
			} {
		image.bindMemory(*memory, 0);
		view = vk::raii::ImageView {
			device,
			vk::ImageViewCreateInfo {
				{}, *image, vk::ImageViewType::e2D,
				format, {}, { aspect_mask, 0, 1, 0, 1 }
			}
		};
	}

	ImageData(std::nullptr_t) {}
};

// Buffer data wrapper
struct BufferData {
	vk::DeviceSize		size;
	vk::BufferUsageFlags	flags;
	vk::MemoryPropertyFlags	memory_properties;

	vk::raii::Buffer	buffer = nullptr;
	vk::raii::DeviceMemory	memory = nullptr;

	// Constructors
	BufferData(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::DeviceSize &size,
			vk::BufferUsageFlags usage,
			vk::MemoryPropertyFlags memory_properties)
			: size { size },
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
	void upload(const std::vector <T> &data) const {
		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			"Buffer data must be host coherent and host visible"
		);

		KOBRA_ASSERT(
			data.size() * sizeof(T) <= size,
			"Buffer size is smaller than data size"
		);

		// Upload data
		void *ptr = memory.mapMemory(0, size);
		memcpy(ptr, data.data(), data.size() * sizeof(T));
		memory.unmapMemory();
	}

	template <class T>
	void upload(const T *const data, const vk::DeviceSize &size) const {
		// Assertions
		KOBRA_ASSERT(
			(memory_properties & vk::MemoryPropertyFlagBits::eHostCoherent)
				&& (memory_properties & vk::MemoryPropertyFlagBits::eHostVisible),
			"Buffer data must be host coherent and host visible"
		);

		KOBRA_ASSERT(
			size <= this->size,
			"Buffer size is smaller than data size"
		);

		// Upload data
		void *ptr = memory.mapMemory(0, size);
		memcpy(ptr, data, size);
		memory.unmapMemory();
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
};

// Device address
inline vk::DeviceAddress buffer_addr(const vk::raii::Device &device, const BufferData &bd)
{
	VkBufferDeviceAddressInfo info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = *bd.buffer
	};

	return vkGetBufferDeviceAddressKHR(*device, &info);
}

// Copy data to an image
void copy_data_to_image(const vk::raii::CommandBuffer &,
		const vk::raii::Buffer &,
		const vk::raii::Image &,
		const vk::Format &,
		uint32_t, uint32_t);

// Create ImageData object from a file
ImageData make_texture(const vk::raii::CommandBuffer &,
		const vk::raii::PhysicalDevice &,
		const vk::raii::Device &,
		BufferData &,
		const std::string &,
		vk::ImageTiling,
		vk::ImageUsageFlags,
		vk::MemoryPropertyFlags,
		vk::ImageAspectFlags);

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
	vk::AttachmentDescription color_attachment {
		{}, format,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::ePresentSrcKHR
	};

	// Add color attachment to list
	attachments.push_back(color_attachment);

	// Create depth attachment
	if (depth_format != vk::Format::eUndefined) {
		vk::AttachmentDescription depth_attachment {
			{}, depth_format,
			vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eDontCare,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined,
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
			{}, max_sets, pool_sizes
		}
	);
}

// Create a descriptor set layout
using DSLB = std::tuple <uint32_t, vk::DescriptorType, uint32_t, vk::ShaderStageFlagBits>;

// TODO: is this function even required? 1:1 parameter mapping
inline vk::raii::DescriptorSetLayout make_descriptor_set_layout
		(const vk::raii::Device &device,
		const std::vector <DSLB> &bindings)
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
			{}, layout_bindings
		}
	);
}

// Create a graphics pipeline
struct GraphicsPipelineInfo {
	const vk::raii::Device &device;
	const vk::raii::RenderPass &render_pass;

	const vk::raii::ShaderModule &vertex_shader;
	const vk::SpecializationInfo *vertex_specialization = nullptr;

	const vk::raii::ShaderModule &fragment_shader;
	const vk::SpecializationInfo *fragment_specialization = nullptr;

	const vk::VertexInputBindingDescription &vertex_binding;
	const std::vector <vk::VertexInputAttributeDescription> &vertex_attributes;

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

	// Vertex input state
	vk::PipelineVertexInputStateCreateInfo vertex_input_info {
		{}, 1, &info.vertex_binding,
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
