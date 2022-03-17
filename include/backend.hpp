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
#include <vulkan/vulkan_core.h>

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

// Extra aliases
using DSLBinding = VkDescriptorSetLayoutBinding;

using VertexBinding = VkVertexInputBindingDescription;
using VertexAttribute = VkVertexInputAttributeDescription;

// TODO: aux class which stores device and physcial device
// (and other per device objects)
class Vulkan {
public:
	///////////////////////
	// Public structures //
	///////////////////////

	// Buffer structure
	struct Buffer {
		VkBuffer	buffer;
		VkDeviceMemory	memory;
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
		VkPipeline pipeline;
		VkPipelineLayout layout;
	};

	// Pipeline creation structure
	template <size_t N>
	struct PipelineInfo {
		Swapchain				swapchain;

		VkRenderPass				render_pass;

		VkShaderModule				vert;
		VkShaderModule				frag;

		std::vector <VkDescriptorSetLayout>	dsls;

		VertexBinding				vertex_binding;
		std::array <VertexAttribute, N>		vertex_attributes;

		size_t					push_consts;
		VkPushConstantRange *			push_consts_range;

		VkPrimitiveTopology			topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

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
		VkShaderModule make_shader(const std::string &path) {
			Glob g = vk->_read_file(path);
			return vk->_mk_shader_module(device, g);
		}

		// Create multiple shader modules
		std::vector <VkShaderModule> make_shaders(const std::vector <std::string> &paths) {
			std::vector <VkShaderModule> modules;
			modules.reserve(paths.size());
			for (const auto &path : paths)
				modules.push_back(make_shader(path));
			return modules;
		}

		// Create a graphics pipeline
		template <size_t N>
		Pipeline make_pipeline(const PipelineInfo <N> &info) const {
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

			// Vertex input
			// auto binding_description = gui::Vertex::vertex_binding();
			// auto attribute_descriptions = gui::Vertex::vertex_attributes();

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
				.cullMode = VK_CULL_MODE_BACK_BIT,
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
				.pDepthStencilState = nullptr,
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
				throw std::runtime_error("failed to create graphics pipeline!");
			}

			Logger::ok("[profiler] Pipeline created");
			return {pipeline, pipeline_layout};
		}
	};

	// Aliases
	using Glob = std::vector <char>;

	// TODO: depreciate this version
	using CommandBufferMaker = std::function <void (const Vulkan *, size_t)>;
	using DeletionTask = std::function <void (Vulkan *)>;	// TODO: Is this Vulkan object needed?

	using DS = VkDescriptorSet;
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
		// TODO: make customizable
		/* for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR; */
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

		if (enable_validation_layers)
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

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
		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
			Logger::error() << "[Vulkan Validation Layer] "
				<< pCallbackData->pMessage << std::endl;

#ifdef KOBRA_THROW_ERROR

			throw std::runtime_error("[Vulkan Validation Layer] "
				"An error occured in the validation layer");

#endif

		} else {

#ifndef KOBRA_VALIDATION_ERROR_ONLY

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

	// Extra initialization
	void init_imgui() {
		// TODO: init on need basis?
		// variable imgui_init = false...
		ImGui::CreateContext();
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

	ImGuiContext init_imgui_glfw(const VkPhysicalDevice &phdev, const Device &device, const Surface &surface, const Swapchain &swapchain) {
		// Context to return
		ImGuiContext context;

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

		VkResult result = vkCreateDescriptorPool(
			device.device, &pool_info,
			nullptr, &context.descriptor_pool
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to create descriptor pool");
			throw (-1);
		}

		// Create render pass
		context.render_pass = make_render_pass(
			device, swapchain,
			VK_ATTACHMENT_LOAD_OP_LOAD,
			VK_ATTACHMENT_STORE_OP_STORE
		);

		// Initialize the ImGui context
		ImGui::CreateContext();

		// Initialize the ImGui for Vulkan renderer
		ImGui_ImplGlfw_InitForVulkan(surface.window, true);

		//this initializes imgui for Vulkan
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = instance;
		init_info.PhysicalDevice = phdev;
		init_info.Device = device.device;
		init_info.Queue = device.graphics_queue;
		init_info.DescriptorPool = context.descriptor_pool;
		init_info.MinImageCount = 2;
		init_info.ImageCount = 2;
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

		ImGui_ImplVulkan_Init(&init_info, context.render_pass);

		// Create command pool
		context.command_pool = make_command_pool(
			phdev, surface, device,
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
			| VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
		);

		// Start a new, single use command buffer
		// TODO: make a method for immediate command buffers
		VkCommandBufferAllocateInfo alloc_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = context.command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkCommandBuffer tmp_cmd_buffer;
		result = vkAllocateCommandBuffers(
			device.device, &alloc_info, &tmp_cmd_buffer
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to allocate command buffer");
			throw (-1);
		}

		// Start recording the command buffer
		VkCommandBufferBeginInfo begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};

		result = vkBeginCommandBuffer(tmp_cmd_buffer, &begin_info);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to begin command buffer");
			throw (-1);
		}

		// Create font textures
		ImGui_ImplVulkan_CreateFontsTexture(tmp_cmd_buffer);

		// End recording the command buffer
		result = vkEndCommandBuffer(tmp_cmd_buffer);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to end command buffer");
			throw (-1);
		}

		// Submit the command buffer
		VkSubmitInfo submit_info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &tmp_cmd_buffer
		};

		result = vkQueueSubmit(device.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan-ImGui] Failed to submit command buffer");
			throw (-1);
		}

		// Wait for the command buffer to finish
		vkQueueWaitIdle(device.graphics_queue);

		// Destroy the command buffer
		// TODO: deletion queue function
		vkFreeCommandBuffers(device.device, context.command_pool, 1, &tmp_cmd_buffer);

		//clear font textures from cpu data
		ImGui_ImplVulkan_DestroyFontUploadObjects();

		// Create command buffer and render pass
		context.command_buffer = make_command_buffer(device, context.command_pool);
		context.semaphore = make_semaphore(device);
		context.fence = make_fence(device);

		// Log the ImGui context creation
		Logger::ok() << "[Vulkan-ImGui] ImGui context created (Surface=" << &surface << ")\n";
		return context;
	}

	// TODO: another overload
	/* void init_imgui() {
		_init_imgui();
		Logger::ok("[Vulkan] ImGui initialized");
	} */

	// Destructor tasks
	void push_deletion_task(const DeletionTask &task) {
		_deletion_tasks.push_back(task);
	}

	// TODO: pop deletion tasks


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
	VkRenderPass make_render_pass(const Device &device,
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
		Logger::ok() << "[Vulkan] Surface created (VkSurfaceKHR="
			<< new_surface << ")\n";

		return Surface {window, new_surface};
	}

	// Create a swapchain and related functions
	Swapchain make_swapchain(const VkPhysicalDevice &, const Device &device, const Surface &);
	void make_framebuffers(const Device &, Swapchain &, VkRenderPass) const;

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
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes),
			.poolSizeCount = (uint32_t) IM_ARRAYSIZE(pool_sizes),
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
		Logger::ok() << "[Vulkan] Descriptor pool created (VkDescriptorPool="
			<< new_descriptor_pool << ")\n";

		return new_descriptor_pool;
	}

	// Create a descriptor set layout
	VkDescriptorSetLayout make_descriptor_set_layout(const Device &device,
			const std::vector <VkDescriptorSetLayoutBinding> &bindings,
			VkAllocationCallbacks *allocator = nullptr) const {
		// Descriptor set layout to return
		VkDescriptorSetLayout new_descriptor_set_layout = VK_NULL_HANDLE;

		// Create info
		VkDescriptorSetLayoutCreateInfo layout_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
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
		Logger::ok() << "[Vulkan] Descriptor set layout created (VkDescriptorSetLayout="
			<< new_descriptor_set_layout << ")\n";
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

	// Static member variables
	static const std::vector <const char *> device_extensions;
	static const std::vector <const char *> validation_layers;
};

#endif
