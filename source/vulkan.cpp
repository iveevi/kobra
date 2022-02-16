#include "../include/backend.hpp"

// Static member variables
const std::vector <const char *> Vulkan::device_extensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

const std::vector <const char *> Vulkan::validation_layers = {
	"VK_LAYER_KHRONOS_validation"
};

////////////////////
// Public methods //
////////////////////

/* Draw a frame
void Vulkan::frame()
{
	// Wait for the next image in the swap chain
	vkWaitForFences(
		device, 1,
		&in_flight_fences[current_frame],
		VK_TRUE, UINT64_MAX
	);

	// Acquire the next image from the swap chain
	uint32_t image_index;
	VkResult result = vkAcquireNextImageKHR(
		device, swch, UINT64_MAX,
		image_available_semaphores[current_frame],
		VK_NULL_HANDLE, &image_index
	);

	// Check if the swap chain is no longer valid
	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		_remk_swapchain();
		return;
	} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		Logger::error("[Vulkan] Failed to acquire swap chain image!");
		throw (-1);
	}

	// Check if the image is being used by the current frame
	if (images_in_flight[image_index] != VK_NULL_HANDLE) {
		vkWaitForFences(
			device, 1,
			&images_in_flight[image_index],
			VK_TRUE, UINT64_MAX
		);
	}

	// Mark the image as in use by this frame
	images_in_flight[image_index] = in_flight_fences[current_frame];

	// Fill out imgui command buffer and render pass
	begin_command_buffer(imgui_cmd_buffer);

	// Begin the render pass
	begin_render_pass(
		imgui_cmd_buffer,
		swch_framebuffers[image_index],
		imgui_render_pass,
		swch_extent,
		0, nullptr
	);

		// ImGui new frame
		// TODO: method
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Show render monitor
		ImGui::Begin("Render Monitor");
		ImGui::Text("fps: %.1f", ImGui::GetIO().Framerate);
		// ImGui::TreeNode("Profiler");
		ImGui::End();

		// ImGui::ShowDemoWindow();

		ImGui::EndFrame();
		ImGui::Render();

		// Render ImGui
		ImGui_ImplVulkan_RenderDrawData(
			ImGui::GetDrawData(),
			imgui_cmd_buffer
			// vulkan.command_buffers[0]
		);

	// End the render pass
	end_render_pass(imgui_cmd_buffer);

	// End the command buffer
	end_command_buffer(imgui_cmd_buffer);

	// Frame submission and synchronization info
	VkSemaphore wait_semaphores[] = {
		image_available_semaphores[current_frame]
	};

	VkPipelineStageFlags wait_stages[] = {
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
	};

	VkSemaphore signal_semaphores[] = {
		render_finished_semaphores[current_frame],
		imgui_semaphore
	};

	// Create information
	// TODO: method
	VkSubmitInfo submit_info {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = wait_semaphores,
		.pWaitDstStageMask = wait_stages,

		.commandBufferCount = 1,
		.pCommandBuffers = &command_buffers[image_index],

		.signalSemaphoreCount = 1,
		.pSignalSemaphores = signal_semaphores
	};

	// Submit the command buffer
	vkResetFences(device, 1, &in_flight_fences[current_frame]);
	
	result = vkQueueSubmit(
		graphics_queue, 1, &submit_info,
		in_flight_fences[current_frame]
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to submit draw command buffer!");
		throw (-1);
	}

	// Wait for the first command buffer to finish
	vkQueueWaitIdle(graphics_queue);

	// Submit ImGui command buffer
	// submit_info.pWaitSemaphores = &imgui_semaphore;
	// submit_info.pWaitDstStageMask = &wait_stages[1];
	submit_info.waitSemaphoreCount = 0;
	// submit_info.signalSemaphoreCount = 0;
	submit_info.pSignalSemaphores = &imgui_semaphore;
	submit_info.pCommandBuffers = &imgui_cmd_buffer;

	// Submit the command buffer
	vkResetFences(device, 1, &imgui_fence);

	result = vkQueueSubmit(
		graphics_queue, 1, &submit_info,
		imgui_fence
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to submit draw command buffer!");
		throw (-1);
	}
	
	// Present the image to the swap chain
	VkSwapchainKHR swchs[] = {swch};
	
	VkPresentInfoKHR present_info {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 2,
		.pWaitSemaphores = signal_semaphores,

		.swapchainCount = 1,
		.pSwapchains = swchs,
		.pImageIndices = &image_index,
		.pResults = nullptr
	};

	result = vkQueuePresentKHR(present_queue, &present_info);
	if (result == VK_ERROR_OUT_OF_DATE_KHR
			|| result == VK_SUBOPTIMAL_KHR
			|| framebuffer_resized) {
		framebuffer_resized = false;
		_remk_swapchain();
	} else if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to present swap chain image!");
		throw (-1);
	}

	// Get the next frame index
	current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
} */

////////////////////
// Buffer methods //
////////////////////

// Allocate a buffer
// TODO: pass buffer propreties as a struct
void Vulkan::make_buffer(const VkPhysicalDevice &phdev,
		const Device &device,
		Buffer &bf,
		size_t size,
		VkBufferUsageFlags usage)
{
	// Buffer creation info
	VkBufferCreateInfo buffer_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};

	VkResult result = vkCreateBuffer(
		device.device, &buffer_info,
		nullptr, &bf.buffer
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to create buffer!");
		// TODO: other exception class
		throw (-1);
	}

	// Allocate memory for the buffer
	VkMemoryRequirements mem_reqs;
	vkGetBufferMemoryRequirements(device.device, bf.buffer, &mem_reqs);

	VkMemoryAllocateInfo alloc_info {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = _find_memory_type(
			phdev,
			mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		)
	};

	// Create device memory
	result = vkAllocateMemory(
		device.device, &alloc_info,
		nullptr, &bf.memory
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to allocate buffer memory!");
		throw (-1);
	}

	// Bind the buffer to the memory
	vkBindBufferMemory(device.device, bf.buffer, bf.memory, 0);

	// Set the buffer properties
	bf.size = size;
	bf.offset = 0;

	// Log creation
	Logger::ok() << "[Vulkan] Buffer created (VkBuffer="
		<< bf.buffer << ", VkDeviceMemory=" << bf.memory
		<< ", size=" << size << ")\n";
}

// Destroy a buffer
void Vulkan::destroy_buffer(const Device &device, Buffer &bf)
{
	vkDestroyBuffer(device.device, bf.buffer, nullptr);
	vkFreeMemory(device.device, bf.memory, nullptr);
}

// Map data to a buffer
void Vulkan::map_buffer(const Device &device, Buffer *buffer, void *data, size_t size)
{
	// Map the buffer
	VkResult result = vkMapMemory(
		device.device, buffer->memory, 0,
		size, 0, &buffer->mapped
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to map buffer memory!");
		throw (-1);
	}

	// Copy the data
	memcpy(buffer->mapped, data, size);

	// Unmap the buffer
	vkUnmapMemory(device.device, buffer->memory);
}

///////////////////////
// Swapchain methods //
///////////////////////

// Fill image views
void Vulkan::_make_image_views(const Device &device, Swapchain &swch) const
{
	// Resize first
	swch.image_views.resize(swch.images.size());

	// Fill with new image views
	for (size_t i = 0; i < swch.images.size(); i++) {
		// Creation info
		VkImageViewCreateInfo create_info {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swch.images[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = swch.image_format,
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
			device.device, &create_info,
			nullptr, &swch.image_views[i]
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create image view!");
			throw (-1);
		}
	}
}

// Create framebuffers
void Vulkan::make_framebuffers(const Device &device, Swapchain &swch, VkRenderPass rpass) const
{
	// Resize first
	swch.framebuffers.resize(swch.image_views.size());

	// Fill with new framebuffers
	for (size_t i = 0; i < swch.image_views.size(); i++) {
		// Arrange attachments
		VkImageView attachments[] = {
			swch.image_views[i]
		};

		// Create framebuffer
		VkFramebufferCreateInfo framebuffer_info {
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = rpass,
			.attachmentCount = 1,
			.pAttachments = attachments,
			.width = swch.extent.width,
			.height = swch.extent.height,
			.layers = 1
		};

		// Safely create framebuffer
		VkResult result = vkCreateFramebuffer(
			device.device, &framebuffer_info,
			nullptr, &swch.framebuffers[i]
		);

		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to create framebuffer!");
			throw(-1);
		}
	}
}

// TODO: should pass number of frames in flight
Vulkan::Swapchain Vulkan::make_swapchain(
		const VkPhysicalDevice &phdev,
		const Device &device,
		const Surface &surface)
{
	// Object to return
	Swapchain swch;

	SwapchainSupport swch_support = _query_swch_support(phdev, surface);

	// Select swapchain properties
	VkSurfaceFormatKHR surface_format = _choose_swch_surface_format(
		swch_support.formats
	);

	VkPresentModeKHR present_mode = _choose_swch_present_mode(
		swch_support.present_modes
	);

	VkExtent2D extent = _choose_swch_extent(
		swch_support.capabilities
	);

	// Set image count
	uint32_t image_count = swch_support.capabilities.minImageCount + 1;
	if (swch_support.capabilities.maxImageCount > 0
			&& image_count > swch_support.capabilities.maxImageCount)
		image_count = swch_support.capabilities.maxImageCount;

	// Build creation info
	// TODO: clean up this part
	VkSwapchainCreateInfoKHR create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	create_info.surface = surface.surface;

	create_info.minImageCount = image_count;
	create_info.imageFormat = surface_format.format;
	create_info.imageColorSpace = surface_format.colorSpace;
	create_info.imageExtent = extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
		| VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	QueueFamilyIndices indices = _find_queue_families(phdev, surface);
	uint32_t queueFamilyIndices[] = {indices.graphics.value(), indices.present.value()};

	if (indices.graphics != indices.present) {
		create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		create_info.queueFamilyIndexCount = 2;
		create_info.pQueueFamilyIndices = queueFamilyIndices;
	} else {
		create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	create_info.preTransform = swch_support.capabilities.currentTransform;
	create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	create_info.presentMode = present_mode;
	create_info.clipped = VK_TRUE;

	// Create the swapchain
	VkResult result = vkCreateSwapchainKHR(
		device.device, &create_info,
		nullptr, &swch.swch
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to create swapchain!");
		throw (-1);
	}

	// Resize images
	vkGetSwapchainImagesKHR(
		device.device, swch.swch,
		&image_count, nullptr
	);

	swch.images.resize(image_count);

	vkGetSwapchainImagesKHR(
		device.device, swch.swch,
		&image_count, swch.images.data()
	);

	// Set other properties
	swch.image_format = surface_format.format;
	swch.extent = extent;

	// Fill out image views
	_make_image_views(device, swch);

	// Log creation and return
	Logger::ok() << "[Vulkan] Swapchain created (VkSwapchain="
		<< swch.swch << ")" << std::endl;

	return swch;
}

/////////////////////////
// Render pass methods //
/////////////////////////

// Create render pass
VkRenderPass Vulkan::make_render_pass(const Device &device,
		const Swapchain &swch,
		VkAttachmentLoadOp load_op,
		VkAttachmentStoreOp store_op,
		VkImageLayout initial_layout,
		VkImageLayout final_layout) const
{
	// Render pass to return
	VkRenderPass new_render_pass = VK_NULL_HANDLE;

	// Create attachment description
	VkAttachmentDescription color_attachment {
		.format = swch.image_format,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = load_op,
		.storeOp = store_op,
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
		device.device, &render_pass_info,
		nullptr, &new_render_pass
	);

	if (result != VK_SUCCESS) {
		Logger::error("[Vulkan] Failed to create render pass!");
		throw(-1);
	}

	// Log creation
	Logger::ok() << "[Vulkan] Render pass created (VkRenderPass="
		<< new_render_pass << ")\n";

	return new_render_pass;
}

// Start and end a render pass
void Vulkan::begin_render_pass(VkCommandBuffer cmd_buffer,
		VkFramebuffer framebuffer,
		VkRenderPass render_pass,
		VkExtent2D extent,
		uint32_t clear_count,
		VkClearValue *clear_values) const
{
	// Begin the render pass
	VkRenderPassBeginInfo render_pass_begin_info {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = render_pass,
		.framebuffer = framebuffer,
		.renderArea = {
			.offset = { 0, 0 },
			.extent = extent
		},
		.clearValueCount = clear_count,
		.pClearValues = clear_values
	};

	vkCmdBeginRenderPass(cmd_buffer, &render_pass_begin_info,
		VK_SUBPASS_CONTENTS_INLINE);
}

void Vulkan::end_render_pass(VkCommandBuffer cmd_buffer) const
{
	vkCmdEndRenderPass(cmd_buffer);
}

/////////////
// Getters //
/////////////

VkPhysicalDeviceProperties Vulkan::phdev_props(const VkPhysicalDevice &phdev) const
{
	// Create the properties struct
	VkPhysicalDeviceProperties props;

	// Get the properties
	vkGetPhysicalDeviceProperties(phdev, &props);

	// Return the properties
	return props;
}

///////////////////
// Other methods //
///////////////////

void Vulkan::idle(const Device &device) const
{
	vkDeviceWaitIdle(device.device);
}