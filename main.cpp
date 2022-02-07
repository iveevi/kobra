#define MERCURY_VALIDATION_LAYERS
#define MERCURY_VALIDATION_ERROR_ONLY
// #define MERCURY_THROW_ERROR

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Standard headers
#include <iostream>
#include <cstring>

// Engine headers
#include "include/backend.hpp"
#include "include/light.hpp"
#include "include/object.hpp"
#include "include/camera.hpp"
#include "include/logger.hpp"

Camera camera {
        Transform {
		glm::vec3(0.0f, 0.0f, -4.0f)
	},
 
        Tunings {
                45.0f,
                800,
                600
        }
};

// Pixel buffer 
Vulkan::Buffer pixel_buffer;
Vulkan::Buffer world_buffer;
Vulkan::Buffer object_buffer;
Vulkan::Buffer light_buffer;

// Compute shader
VkShaderModule compute_shader;

// Command buffer function per frame index
void cmd_buffer_maker(Vulkan *vk, size_t i) {	
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
		400, 400, 1
	);
}

void descriptor_set_maker(Vulkan *vulkan, int i)
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
		.buffer = object_buffer.buffer,
		.offset = 0,
		.range = object_buffer.size
	};

	VkDescriptorBufferInfo lb_info {
		.buffer = light_buffer.buffer,
		.offset = 0,
		.range = light_buffer.size
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

	VkWriteDescriptorSet writes[] = {
		pb_write,
		wb_write,
		ob_write,
		lb_write
	};

	vkUpdateDescriptorSets(
		vulkan->device, 4,
		&writes[0],
		0, nullptr
	);
}

// TODO: rename and initializer list constructor
struct alignas(16) Aligned {
	glm::vec4 data;

	Aligned() {}
	Aligned(const glm::vec4 &d) : data(d) {}
};

// World structure
// TODO: refactor to GPU world,
// store cpu world in another structure that is called World
struct World {
	uint32_t objects;
	uint32_t lights;
	uint32_t backgound;

	uint32_t width = 800;
	uint32_t height = 600;

	alignas(16) glm::vec3 camera_position = glm::vec3(0.0f, 0.0f, -15.0f);
	alignas(16) glm::vec3 camera_forward = glm::vec3(0.0f, 0.0f, 1.0f);
	alignas(16) glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
	alignas(16) glm::vec3 camera_right = glm::vec3(1.0f, 0.0f, 0.0f);

	float fov = camera.tunings.fov;
	float scale = camera.tunings.scale;
	float aspect = camera.tunings.aspect;

	// TODO: store indices for objects and lights
};

World world = {
	.objects = 6,
	.lights = 1,
	.backgound = 0x202020,
};

// Keyboard callback
// TODO: in class
bool rerender = true;
bool mouse_tracking = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

        // Camera movement
        float speed = 0.5f;
        if (key == GLFW_KEY_W) {
                camera.transform.position += camera.transform.forward * speed;
                rerender = true;
        } else if (key == GLFW_KEY_S) {
                camera.transform.position -= camera.transform.forward * speed;
                rerender = true;
        }

        if (key == GLFW_KEY_A) {
                camera.transform.position -= camera.transform.right * speed;
                rerender = true;
        } else if (key == GLFW_KEY_D) {
                camera.transform.position += camera.transform.right * speed;
                rerender = true;
        }

	if (key == GLFW_KEY_E) {
		camera.transform.position += camera.transform.up * speed;
		rerender = true;
	} else if (key == GLFW_KEY_Q) {
		camera.transform.position -= camera.transform.up * speed;
		rerender = true;
	}

	// Tab to toggle cursor visibility
	static bool cursor_visible = false;
	if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
		cursor_visible = !cursor_visible;
		mouse_tracking = !mouse_tracking;
		glfwSetInputMode(window, GLFW_CURSOR, cursor_visible ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
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

	camera.transform.set_euler(pitch, yaw);

	last_x = xpos;
	last_y = ypos;
	
	rerender = true;
}

// ImGui command buffer and render pass
/* TODO: class objects
VkCommandBuffer imgui_command_buffer;
VkRenderPass imgui_render_pass; */

int main()
{
	// Initialize Vulkan
	Vulkan vulkan;

	/* Create imgui cmd buffer and render pass
	imgui_command_buffer = vulkan.make_command_buffer();
	imgui_render_pass = vulkan.make_render_pass(
		VK_ATTACHMENT_LOAD_OP_LOAD,
		VK_ATTACHMENT_STORE_OP_STORE
	); */

	// Pixel initialization
	Aligned *objects = new Aligned[16] {
		{ {-1.0f, 0.0f, 4.0f, 1.0f} },
		{ {0.1f, 0.5f, 0.2f, 0.0f} },
		
		{ {0.5f, 5.0f, 3.0f, 3.0f } },
		{ {0.9f, 0.5f, 0.2f, 0.0f} },
		
		{ {6.0f, -2.0f, 5.0f, 6.0f } },
		{ {0.4f, 0.4f, 0.9f, 0.0f} },
		
		{ {6.0f, 3.0f, 11.5f, 2.0f} },
		{ {0.5f, 0.1f, 0.6f, 0.0f} },
		
		{ {6.0f, 3.0f, -2.0f, 2.0f} },
		{ {0.6f, 0.5f, 0.3f, 0.0f} },

		{ {0.0f, 0.0f, 0.0f, 0.2f} },
		{ {1.0f, 0.5f, 1.0f, 1.0f} }
	};

	Aligned *lights = new Aligned[16] {
		{ {0.0f, 0.0f, 0.0f, 1.0f} }
	};

	// Pixel data
	size_t pixel_size = 4 * sizeof(char) * 800 * 600;
	size_t world_size = sizeof(World);
	size_t object_size = sizeof(Aligned) * 16;
	size_t light_size = sizeof(Aligned) * 16;

	VkBufferUsageFlags pixel_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags world_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags object_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags light_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Create buffers
	vulkan.make_buffer(pixel_buffer, pixel_size, pixel_usage);
	vulkan.make_buffer(world_buffer, sizeof(World), world_usage);
	vulkan.make_buffer(object_buffer, object_size, object_usage);
	vulkan.make_buffer(light_buffer, light_size, light_usage);

	vulkan.map_buffer(&world_buffer, &world, sizeof(World));
	vulkan.map_buffer(&object_buffer, objects, object_size);
	vulkan.map_buffer(&light_buffer, lights, light_size);

	// Add all buffers to deletion queue
	vulkan.push_deletion_task(
		[&](Vulkan *vk) {
			vk->destroy_buffer(pixel_buffer);
			vk->destroy_buffer(world_buffer);
			vk->destroy_buffer(object_buffer);
			vk->destroy_buffer(light_buffer);
			Logger::ok("[main] Deleted buffers");
		}
	);

	// Compute shader descriptor
	compute_shader = vulkan.make_shader("shaders/pixel.spv");

	// Add shader to deletion queue'
	vulkan.push_deletion_task(
		[&](Vulkan *vk) {
			vkDestroyShaderModule(vk->device, compute_shader, nullptr);
			Logger::ok("[main] Deleted compute shader");
		}
	);

	// Buffer descriptor
	for (int i = 0; i < vulkan.swch_images.size(); i++)
		descriptor_set_maker(&vulkan, i);

	// Set keyboard callback
	glfwSetKeyCallback(vulkan.window, key_callback);

	// Mouse options
	glfwSetInputMode(vulkan.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set mouse callback
	glfwSetCursorPosCallback(vulkan.window, mouse_callback);
	
	vulkan.set_command_buffers(cmd_buffer_maker);

	// ImGui and IO
	vulkan.init_imgui();
	ImGuiIO &io = ImGui::GetIO();

	// Main render loop
	float time = 0.0f;
	float delta_time = 0.01f;
	while (!glfwWindowShouldClose(vulkan.window)) {
		glfwPollEvents();

		// Update world data
		world.camera_position = camera.transform.position;
		world.camera_forward = camera.transform.forward;
		world.camera_right = camera.transform.right;
		world.camera_up = camera.transform.up;
		vulkan.map_buffer(&world_buffer, &world, sizeof(World));

		// Update lights
		float amplitude = 5.0f;
		lights[0] = glm::vec4 {
			amplitude * sin(time), 5.0f,
			amplitude * cos(time), 1.0f
		};

		vulkan.map_buffer(&light_buffer, lights, light_size);

		// Show frame
		vulkan.frame();

		// Time
		time += delta_time;
	}

	vulkan.idle();
}
