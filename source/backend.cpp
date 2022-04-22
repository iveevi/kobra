#include "../include/backend.hpp"

namespace kobra {

// Get (or create) the singleton context
const vk::raii::Context &get_vulkan_context()
{
	// Global context
	static vk::raii::Context context;
	return context;
}

}
