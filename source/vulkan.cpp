#include "../include/backend.hpp"

////////////////////
// Public methods //
////////////////////

// Draw a frame
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
		ImGui::TreeNode("Profiler");
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
}

/////////////
// Getters //
/////////////

VkPhysicalDeviceProperties Vulkan::phdev_props() const
{
	// Create the properties struct
	VkPhysicalDeviceProperties props;

	// Get the properties
	vkGetPhysicalDeviceProperties(physical_device, &props);

	// Return the properties
	return props;
}
