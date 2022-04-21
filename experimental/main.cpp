// Standard headers
#include <iostream>

// More Vulkan stuff
#include <vulkan/vulkan_core.h>

#define KOBRA_VALIDATION_LAYERS

// Engine headers
#include "../include/backend.hpp"
#include "../include/camera.hpp"
#include "../shaders/raster/bindings.h"

using namespace kobra;

// Cube vertices
float cube_vertices[] = {
	// positions          // texture coords
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
	0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

	-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
};

uint32_t cube_indices[] = {
	0, 1, 2,
	0, 2, 3,
	4, 5, 6,
	4, 6, 7,
	8, 9, 10,
	8, 10, 11,
	12, 13, 14,
	12, 14, 15,
	16, 17, 18,
	16, 18, 19,
	20, 21, 22,
	20, 22, 23,
};

int main()
{
	// Camera
	Camera camera = Camera {
		Transform {
			{0, 0, 5},
			{-0.2, 0, 0}
		},

		Tunings { 45.0f, 800, 800 }
	};

	// Choosing physical device
	auto window = Window("Vulkan RT", {200, 200});
	vk::raii::SurfaceKHR surface = make_surface(window);

	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		"VK_KHR_ray_tracing_pipeline"
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	auto phdev = pick_physical_device(predicate);

	std::cout << "Chosen device: " << phdev.getProperties().deviceName << std::endl;
	std::cout << "\tgraphics queue family: " << find_graphics_queue_family(phdev) << std::endl;
	std::cout << "\tpresent queue family: " << find_present_queue_family(phdev, surface) << std::endl;

	// Verification and creating a logical device
	auto queue_family = find_queue_families(phdev, surface);
	std::cout << "\tqueue family (G): " << queue_family.graphics << std::endl;
	std::cout << "\tqueue family (P): " << queue_family.present << std::endl;

	auto device = make_device(phdev, queue_family, extensions);

	// Command pool and buffer
	auto command_pool = vk::raii::CommandPool {
		device,
		{
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			queue_family.graphics
		}
	};

	auto command_buffer = make_command_buffer(device, command_pool);

	// Queues
	auto graphics_queue = vk::raii::Queue { device, queue_family.graphics, 0 };
	auto present_queue = vk::raii::Queue { device, queue_family.present, 0 };

	// Swapchain
	auto swapchain = Swapchain {phdev, device, surface, window.extent, queue_family};

	// Depth buffer and render pass
	auto depth_buffer = DepthBuffer {phdev, device, vk::Format::eD32Sfloat, window.extent};
	auto render_pass = make_render_pass(device, swapchain.format, depth_buffer.format);

	auto framebuffers = make_framebuffers(
		device, render_pass,
		swapchain.image_views,
		&depth_buffer.view,
		window.extent
	);

	// Load shaders
	auto vertex = make_shader_module(device, "shaders/bin/raster/vertex.spv");
	auto fragment = make_shader_module(device, "shaders/bin/raster/color_frag.spv");

	// Descriptor set layout and pipeline layout
	auto dsl = make_descriptor_set_layout(device, {
		{
			vk::DescriptorType::eSampler,
			RASTER_BINDING_ALBEDO_MAP,
			vk::ShaderStageFlagBits::eFragment
		},

		{
			vk::DescriptorType::eSampler,
			RASTER_BINDING_NORMAL_MAP,
			vk::ShaderStageFlagBits::eFragment
		},
	});

	auto ppl = vk::raii::PipelineLayout {device, {{}, *dsl}};
}
