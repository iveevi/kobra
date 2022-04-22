#include "../include/backend.hpp"

namespace kobra {

// Get (or create) the singleton instance
const vk::raii::Instance &get_vulkan_instance()
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

	static vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;

	static vk::raii::Instance instance {
		get_vulkan_context(),
		instance_info,
	};

#ifdef KOBRA_VALIDATION_LAYERS

	/* auto dispatcher = vk::DispatchLoaderDynamic {
		*instance,
		vkGetInstanceProcAddr
	};

	auto severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;

	auto type = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

	vk::DebugUtilsMessengerCreateInfoEXT debug_messenger_info {
		vk::DebugUtilsMessengerCreateFlags(),
		severity,
		type,
		debug_logger,
		nullptr
	};

	debug_messenger = instance.createDebugUtilsMessengerEXT(
		debug_messenger_info,
		dispatcher
	); */

#endif

	return instance;
}

}
