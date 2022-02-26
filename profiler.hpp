#ifndef PROFILER_APPLICATION_H_
#define PROFILER_APPLICATION_H_

#include "global.hpp"
#include "include/backend.hpp"
#include "include/buffer_manager.hpp"
#include "include/gui/gui.hpp"
#include <vulkan/vulkan_core.h>

using namespace mercury;

// Profiler application
class ProfilerApplication : public mercury::App {
	VkRenderPass			render_pass;
	VkCommandPool			command_pool;

	std::vector <VkCommandBuffer>	command_buffers;

	VkDescriptorPool		descriptor_pool;

	// Shaders
	VkShaderModule			basic_vert;
	VkShaderModule			basic_frag;

	VkDescriptorSetLayout		descriptor_set_layout;
	VkDescriptorSet			descriptor_set;

	// Sync objects
	std::vector <VkFence>		in_flight_fences;
	std::vector <VkFence>		images_in_flight;

	std::vector <VkSemaphore>	smph_image_available;
	std::vector <VkSemaphore>	smph_render_finished;

	// Vertex buffer
	gui::VertexBuffer		vb;
	gui::IndexBuffer		ib;

	VkPipeline graphics_pipeline;
	void make_pipeline() {
		// Create pipeline stages
		VkPipelineShaderStageCreateInfo vertex {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = basic_vert,
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo fragment {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = basic_frag,
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo shader_stages[] = { vertex, fragment };

		// Vertex input
		auto binding_description = gui::Object::vertex_binding();
		auto attribute_descriptions = gui::Object::vertex_attributes();

		VkPipelineVertexInputStateCreateInfo vertex_input {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &binding_description,
			.vertexAttributeDescriptionCount = static_cast <uint32_t> (attribute_descriptions.size()),
			.pVertexAttributeDescriptions = attribute_descriptions.data()
		};

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo input_assembly {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE
		};

		// Viewport
		VkViewport viewport {
			.x = 0.0f,
			.y = 0.0f,
			.width = (float) swapchain.extent.width,
			.height = (float) swapchain.extent.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f
		};

		// Scissor
		VkRect2D scissor {
			.offset = {0, 0},
			.extent = swapchain.extent
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
			.lineWidth = 1.0f,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE
		};

		// Multisampling
		// TODO: method
		VkPipelineMultisampleStateCreateInfo multisampling {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.sampleShadingEnable = VK_FALSE,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.minSampleShading = 1.0f,
			.pSampleMask = nullptr,
			.alphaToCoverageEnable = VK_FALSE,
			.alphaToOneEnable = VK_FALSE
		};

		// Color blending
		VkPipelineColorBlendAttachmentState color_blend_attachment {
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
			.blendEnable = VK_FALSE
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
			.setLayoutCount = 0,
			.pushConstantRangeCount = 0
		};

		VkPipelineLayout pipeline_layout;
		VkResult result = vkCreatePipelineLayout(
			context.vk_device(),
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
			.renderPass = render_pass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
			.basePipelineIndex = -1
		};

		result = vkCreateGraphicsPipelines(
			context.vk_device(),
			VK_NULL_HANDLE,
			1,
			&pipeline_info,
			nullptr,
			&graphics_pipeline
		);

		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		Logger::ok("[profiler] Graphics pipeline created");
	}

	// Program the command buffer
	void maker(size_t i) {
		// Clear color
		VkClearValue clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };

		// Render pass creation info
		VkRenderPassBeginInfo render_pass_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = swapchain.framebuffers[i],
			.renderArea = {
				.offset = {0, 0},
				.extent = swapchain.extent
			},
			.clearValueCount = 1,
			.pClearValues = &clear_color
		};

		// TODO: use method to being and end render pass
		vkCmdBeginRenderPass(
			command_buffers[i],
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);

		// Bind pipeline
		vkCmdBindPipeline(
			command_buffers[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			graphics_pipeline
		);

		// Bind vertex buffer
		VkBuffer vertex_buffers[] = {vb.vk_buffer()};
		VkDeviceSize offsets[] = {0};

		vkCmdBindVertexBuffers(
			command_buffers[i],
			0,
			1,
			vertex_buffers,
			offsets
		);

		// Bind index buffer
		vkCmdBindIndexBuffer(
			command_buffers[i],
			ib.vk_buffer(),
			0,
			VK_INDEX_TYPE_UINT32
		);

		// Draw
		vkCmdDrawIndexed(
			command_buffers[i],
			ib.size(),
			1, 0, 0, 0
		);

		// TODO: use the methods
		vkCmdEndRenderPass(command_buffers[i]);
	}
public:
	ProfilerApplication(Vulkan *vk) : App({
		.ctx = vk,
		.width = 800,
		.height = 600,
		.name = "Mercury Profiler"
	}) {
		// Create render pass
		render_pass = context.vk->make_render_pass(
			context.device,
			swapchain,
			VK_ATTACHMENT_LOAD_OP_CLEAR,
			VK_ATTACHMENT_STORE_OP_STORE
		);

		// Create framebuffers
		context.vk->make_framebuffers(context.device, swapchain, render_pass);

		// Create command pool
		command_pool = context.vk->make_command_pool(
			context.phdev,
			surface,
			context.device,
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
		);

		// Create descriptor pool
		descriptor_pool = context.vk->make_descriptor_pool(context.device);

		// Create descriptor set layout
		descriptor_set_layout = context.vk->make_descriptor_set_layout(context.device, dsl_bindings);

		// Create descriptor sets
		descriptor_set = context.vk->make_descriptor_set(
			context.device,
			descriptor_pool,
			descriptor_set_layout
		); 

		// Create sync objects
		// TODO: use max frames in flight
		images_in_flight.resize(swapchain.images.size(), VK_NULL_HANDLE);
		for (size_t i = 0; i < 2; i++) {
			in_flight_fences.push_back(context.vk->make_fence(context.device, VK_FENCE_CREATE_SIGNALED_BIT));
			smph_image_available.push_back(context.vk->make_semaphore(context.device));
			smph_render_finished.push_back(context.vk->make_semaphore(context.device));
		}

		// Load shaders
		basic_vert = context.vk->make_shader(context.device, "shaders/bin/gui/basic_vert.spv");
		basic_frag = context.vk->make_shader(context.device, "shaders/bin/gui/basic_frag.spv");

		// Create pipeline
		make_pipeline();

		// Allocate the vertex buffer
		BFM_Settings vb_settings {
			.size = 1024,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		vb = gui::VertexBuffer(context, vb_settings);
		ib = gui::IndexBuffer(context, vb_settings);

		// Create command buffers
		update_command_buffers();
	}

	// Present frame
	void present() {
		// Wait for the next image in the swap chain
		vkWaitForFences(
			context.vk_device(), 1,
			&in_flight_fences[frame_index],
			VK_TRUE, UINT64_MAX
		);

		// Acquire the next image from the swap chain
		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(
			context.vk_device(), swapchain.swch, UINT64_MAX,
			smph_image_available[frame_index],
			VK_NULL_HANDLE, &image_index
		);

		// Check if the swap chain is no longer valid
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			// TODO: recreate swap chain
			// _remk_swapchain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			Logger::error("[Vulkan] Failed to acquire swap chain image!");
			throw (-1);
		}

		// Check if the image is being used by the current frame
		if (images_in_flight[image_index] != VK_NULL_HANDLE) {
			vkWaitForFences(
				context.vk_device(), 1,
				&images_in_flight[image_index],
				VK_TRUE, UINT64_MAX
			);
		}

		// Mark the image as in use by this frame
		images_in_flight[image_index] = in_flight_fences[frame_index];

		// Frame submission and synchronization info
		VkSemaphore wait_semaphores[] = {
			smph_image_available[frame_index]
		};

		VkPipelineStageFlags wait_stages[] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
		};

		VkSemaphore signal_semaphores[] = {
			smph_render_finished[frame_index],
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
		vkResetFences(context.device.device, 1, &in_flight_fences[frame_index]);

		result = vkQueueSubmit(
			context.device.graphics_queue, 1, &submit_info,
			in_flight_fences[frame_index]
		);

		if (result != VK_SUCCESS) {
			Logger::error("[main] Failed to submit draw command buffer!");
			throw (-1);
		}

		// Present the image to the swap chain
		VkSwapchainKHR swchs[] = {swapchain.swch};
		
		VkPresentInfoKHR present_info {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signal_semaphores,
			.swapchainCount = 1,
			.pSwapchains = swchs,
			.pImageIndices = &image_index,
			.pResults = nullptr
		};
		
		result = vkQueuePresentKHR(
			context.device.present_queue,
			&present_info
		);
	
		/* if (result == VK_ERROR_OUT_OF_DATE_KHR
				|| result == VK_SUBOPTIMAL_KHR
				|| framebuffer_resized) {
			framebuffer_resized = false;
			_remk_swapchain();
		} else*/
		
		// TODO: check resizing (in app)
		if (result != VK_SUCCESS) {
			Logger::error("[Vulkan] Failed to present swap chain image!");
			throw (-1);
		}
	}

	// Update geometry
	void update_geometry() {
		vb.reset_push_back();
		ib.reset_push_back();
		
		gui::Rect({0, 0.2}, {0.5, 0.5}, {1.0, 0.5, 0.0}).upload(vb, ib);
		gui::Rect({0.2, -0.2}, {0.6, 0.1}, {0.0, 0.5, 1.0}).upload(vb, ib);
		
		vb.sync_size();
		ib.sync_size();

		vb.upload();
		ib.upload();
	}

	void frame() override {
		// Update geometry
		update_geometry();

		// Present the frame
		present();
	}

	// Update command buffers
	void update_command_buffers() {
		auto ftn = [this](const Vulkan *ctx, size_t i) {
			maker(i);
		};

		// TODO: static methods
		context.vk->set_command_buffers(
			context.device,
			swapchain, command_pool,
			command_buffers,
			ftn
		);
	}

	// Static variables
	static const std::vector <DSLBinding> dsl_bindings;
};

const std::vector <DSLBinding> ProfilerApplication::dsl_bindings {
	/* DSLBinding {
		.binding = 0,
		.descriptor_type = VK_DESCRIPTOR_TYPE_VERTEX_BUFFER,
		.descriptor_count = 1,
		.stage_flags = VK_SHADER_STAGE_VERTEX_BIT
	} */
};

#endif