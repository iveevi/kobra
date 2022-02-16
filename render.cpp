// Engine headers
#include "global.hpp"

// Pixel buffer
Vulkan::Buffer pixel_buffer;
Vulkan::Buffer world_buffer;
Vulkan::Buffer objects_buffer;
Vulkan::Buffer lights_buffer;
Vulkan::Buffer materials_buffer;

// Compute shader
VkShaderModule compute_shader;

/* Command buffer function per frame index
void cmd_buffer_maker(const Vulkan *vk, size_t i) {	
	// Get image at current index
	VkImage image = vk->swch_images[i];

	// Transition image layout to present
	// TODO: method to transition image layout,
	//	includes pipeline barrier functoin exeution
	VkImageMemoryBarrier image_memory_barrier {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};

	vkCmdPipelineBarrier(
		vk->command_buffers[i],
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		0,
		0,
		nullptr,
		0,
		nullptr,
		1,
		&image_memory_barrier
	);

	// Buffer copy regions
	VkBufferImageCopy buffer_copy_region {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {
			vk->swch_extent.width,
			vk->swch_extent.height,
			1
		}
	};

	// Copy buffer to image
	vkCmdCopyBufferToImage(
		vk->command_buffers[i],
		pixel_buffer.buffer,
		image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&buffer_copy_region
	);
	
	// Render pass creation info
	VkRenderPassBeginInfo render_pass_info {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = vk->render_pass,
		.framebuffer = vk->swch_framebuffers[i],
		.renderArea = {
			.offset = {0, 0},
			.extent = vk->swch_extent
		},
		.clearValueCount = 0,
		.pClearValues = nullptr
	};

	// Render pass creation
	vkCmdBeginRenderPass(
		vk->command_buffers[i],
		&render_pass_info,
		VK_SUBPASS_CONTENTS_INLINE
	);

		// Create pipeline
		VkPipelineLayoutCreateInfo pipeline_layout_info {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &vk->descriptor_set_layouts[i],
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};

		VkPipelineLayout pipeline_layout;

		VkResult res = vkCreatePipelineLayout(
			vk->device,
			&pipeline_layout_info,
			nullptr,
			&pipeline_layout
		);

		if (res != VK_SUCCESS) {
			std::cerr << "Failed to create pipeline layout" << std::endl;
			return;
		}

		// Execute compute shader on the pixel buffer
		VkPipeline pipeline;

		VkComputePipelineCreateInfo compute_pipeline_info {
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = compute_shader,
				.pName = "main"
			},
			.layout = pipeline_layout
		};

		res = vkCreateComputePipelines(
			vk->device,
			VK_NULL_HANDLE,
			1,
			&compute_pipeline_info,
			nullptr,
			&pipeline
		);

		if (res != VK_SUCCESS) {
			std::cerr << "Failed to create compute pipeline" << std::endl;
			return;
		}

		// Bind pipeline
		vkCmdBindPipeline(
			vk->command_buffers[i],
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipeline
		);

		// Bind buffer
		vkCmdBindDescriptorSets(
			vk->command_buffers[i],
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipeline_layout,
			0,
			1,
			vk->descriptor_sets.data(),
			0,
			nullptr
		);

	vkCmdEndRenderPass(vk->command_buffers[i]);
		
	// Dispatch
	vkCmdDispatch(
		vk->command_buffers[i],
		1000, 1000, 1
	);
}

void descriptor_set_maker(Vulkan *vulkan, size_t i)
{
	VkDescriptorBufferInfo pb_info {
		.buffer = pixel_buffer.buffer,
		.offset = 0,
		.range = pixel_buffer.size
	};
	
	VkDescriptorBufferInfo wb_info {
		.buffer = world_buffer.buffer,
		.offset = 0,
		.range = world_buffer.size
	};
	
	VkDescriptorBufferInfo ob_info {
		.buffer = objects_buffer.buffer,
		.offset = 0,
		.range = objects_buffer.size
	};

	VkDescriptorBufferInfo lb_info {
		.buffer = lights_buffer.buffer,
		.offset = 0,
		.range = lights_buffer.size
	};
	
	VkDescriptorBufferInfo mt_info {
		.buffer = materials_buffer.buffer,
		.offset = 0,
		.range = materials_buffer.size
	};

	VkWriteDescriptorSet pb_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &pb_info
	};
	
	VkWriteDescriptorSet wb_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 1,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &wb_info
	};
	
	VkWriteDescriptorSet ob_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 2,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &ob_info
	};

	VkWriteDescriptorSet lb_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 3,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &lb_info
	};

	VkWriteDescriptorSet mt_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 4,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &mt_info
	};

	VkWriteDescriptorSet writes[] = {
		pb_write,
		wb_write,
		ob_write,
		lb_write,
		mt_write
	};

	vkUpdateDescriptorSets(
		vulkan->device, 5,
		&writes[0],
		0, nullptr
	);
} */

// Keyboard callback
// TODO: in class
bool mouse_tracking = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

        // Camera movement
        float speed = 0.5f;
        if (key == GLFW_KEY_W)
                world.camera.transform.position += world.camera.transform.forward * speed;
        else if (key == GLFW_KEY_S)
                world.camera.transform.position -= world.camera.transform.forward * speed;

        if (key == GLFW_KEY_A)
                world.camera.transform.position -= world.camera.transform.right * speed;
        else if (key == GLFW_KEY_D)
                world.camera.transform.position += world.camera.transform.right * speed;

	if (key == GLFW_KEY_E)
		world.camera.transform.position += world.camera.transform.up * speed;
	else if (key == GLFW_KEY_Q)
		world.camera.transform.position -= world.camera.transform.up * speed;

	// Tab to toggle cursor visibility
	static bool cursor_visible = false;
	if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
		cursor_visible = !cursor_visible;
		mouse_tracking = !mouse_tracking;
		glfwSetInputMode(
			window,
			GLFW_CURSOR,
			cursor_visible ?
				GLFW_CURSOR_NORMAL
				: GLFW_CURSOR_DISABLED
		);
	}
}

// Mouse movement callback
void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
	static bool first_mouse = true;
	static float last_x = WIDTH / 2.0f;
	static float last_y = HEIGHT / 2.0f;
	static const float sensitivity = 0.001f;

	if (!mouse_tracking)
		return;

	if (first_mouse) {
		first_mouse = false;
		last_x = xpos;
		last_y = ypos;
		return;
	}

	// Store pitch and yaw
	static float pitch = 0.0f;
	static float yaw = 0.0f;

	float xoffset = xpos - last_x;
	float yoffset = ypos - last_y;

	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	else if (pitch < -89.0f)
		pitch = -89.0f;

	world.camera.transform.set_euler(pitch, yaw);

	last_x = xpos;
	last_y = ypos;
}
