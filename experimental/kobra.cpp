#include "kobra.hpp"
#include "imgui.h"
#include "../include/texture.hpp"
#include <vulkan/vulkan_core.h>

// Static member variables

// TODO: cache in constructor or something...
const std::vector <VkDescriptorSetLayoutBinding> RTApp::compute_dsl_bindings = {
	VkDescriptorSetLayoutBinding {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 2,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 3,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 4,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 5,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 6,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 7,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 8,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 9,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	// Texture module
	VkDescriptorSetLayoutBinding {
		.binding = 10,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 11,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},
};

const std::vector <VkDescriptorSetLayoutBinding> RTApp::preproc_dsl_bindings = {
	VkDescriptorSetLayoutBinding {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr
	},
};

// Fill out command buffer
// TODO: do we need the vk parameter?
void RTApp::maker(const Vulkan *vk, size_t i)
{
	// Render pass creation info
	VkRenderPassBeginInfo render_pass_info {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = render_pass,
		.framebuffer = swapchain.framebuffers[i],
		.renderArea = {
			.offset = {0, 0},
			.extent = swapchain.extent
		},
		.clearValueCount = 0,
		.pClearValues = nullptr
	};

	// Render pass creation
	// TODO: use method to being and end render pass
	vkCmdBeginRenderPass(
		command_buffers[i],
		&render_pass_info,
		VK_SUBPASS_CONTENTS_INLINE
	);

		// Create pipeline
		VkPipelineLayoutCreateInfo pipeline_layout_info {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};

		VkPipelineLayout pipeline_layout;

		VkResult res = vkCreatePipelineLayout(
			context.vk_device(),
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
			context.device.device,
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

		//////////////////////////
		// Post process shaders //
		//////////////////////////

		// Append post processing graphics pipeline
		VkPipelineShaderStageCreateInfo vertex {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = pp_vert_shader,
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo fragment {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = pp_frag_shader,
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo shader_stages[] = {vertex, fragment};

		// TODO: clean
                VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
                vertexInputInfo.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                vertexInputInfo.vertexBindingDescriptionCount = 0;
                vertexInputInfo.vertexAttributeDescriptionCount = 0;

                VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
                inputAssembly.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float) swapchain.extent.width;
		viewport.height = (float) swapchain.extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swapchain.extent;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Create pipeline
		VkPipelineLayoutCreateInfo preproc_pli {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &preproc_dsl,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};

		VkPipelineLayout preproc_pl;

		res = vkCreatePipelineLayout(
			context.vk_device(),
			&preproc_pli,
			nullptr,
			&preproc_pl
		);

                VkGraphicsPipelineCreateInfo graphics_pipeline_info {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = shader_stages,
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = nullptr,
			.pColorBlendState = &colorBlending,
			.pDynamicState = nullptr,
			.layout = preproc_pl,
			.renderPass = render_pass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
		};

		// Create graphics pipeline
		VkPipeline graphics_pipeline;
		res = vkCreateGraphicsPipelines(
			context.vk_device(), VK_NULL_HANDLE,
			1, &graphics_pipeline_info,
			nullptr, &graphics_pipeline
		);

		if (res != VK_SUCCESS) {
			Logger::error("[main] Failed to create graphics pipeline");
			throw -1;
		}

		// Bind pipelines
		vkCmdBindPipeline(
			command_buffers[i],
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipeline
		);

		vkCmdBindPipeline(
			command_buffers[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			graphics_pipeline
		);

		// Bind buffers
		vkCmdBindDescriptorSets(
			command_buffers[i],
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipeline_layout,
			0, 1, &descriptor_set,
			0, nullptr
		);

		vkCmdBindDescriptorSets(
			command_buffers[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			preproc_pl,
			0, 1, &preproc_ds,
			0, nullptr
		);

		vkCmdDraw(command_buffers[i], 6, 1, 0, 0);

	// TODO: use the methods
	vkCmdEndRenderPass(command_buffers[i]);

	// Dispatch
	vkCmdDispatch(
		command_buffers[i],
		swapchain.extent.width,
		swapchain.extent.height, 1
	);
}

////////////////////
// Buffer methods //
////////////////////

// Map all the buffers
bool RTApp::map_buffers(Vulkan *vk)
{
	// Reset pushback indices
	_bf_objects.reset_push_back();
	_bf_lights.reset_push_back();
	_bf_materials.reset_push_back();
	_bf_vertices.reset_push_back();
	_bf_transforms.reset_push_back();

	// Resizing and remaking buffers
	int resized = 0;

	// World update
	// TODO: constructor that resets pushback indices,
	// and a method to sync and flush the buffers
	WorldUpdate wu {
		.bf_objs = &_bf_objects,
		.bf_lights = &_bf_lights,
		.bf_mats = &_bf_materials,
		.bf_verts = &_bf_vertices,
		.bf_trans = &_bf_transforms
	};

	// Generate world data and write to buffers
	world.write(wu);

	// Calculate size of world buffer
	size_t world_size = sizeof(rt::GPUWorld)
		+ wu.indices.size() * sizeof(uint);

	// Copy world and indices
	gworld = world.dump();
	gworld.objects = wu.bf_objs->push_size();	// TODO: this is not the correct size

	if (_bf_world.size() < world_size) {
		Logger::notify() << "Resizing world buffer" << std::endl;
		resized += _bf_world.resize(world_size);
	}

	_bf_world.write(
		(const uint8_t *) &gworld,
		sizeof(rt::GPUWorld)
	);

	_bf_world.write(
		(const uint8_t *) wu.indices.data(),
		4 * wu.indices.size(),
		sizeof(rt::GPUWorld)
	);

	resized += _bf_objects.sync_size();
	resized += _bf_lights.sync_size();
	resized += _bf_materials.sync_size();
	resized += _bf_vertices.sync_size();
	resized += _bf_transforms.sync_size();

	// Map buffers
	_bf_world.upload();
	_bf_objects.upload();
	_bf_lights.upload();
	_bf_materials.upload();
	_bf_vertices.upload();
	_bf_transforms.upload();

	// Return true if any buffers were resized
	return (resized > 0);
}

// Allocate buffers
void RTApp::allocate_buffers()
{
	static const VkBufferUsageFlags buffer_usage =
		VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Allocate buffers
	size_t pixels = 800 * 600;

	// Read only buffers
	_bf_pixels = BufferManager <uint> {
		context,
		BFM_Settings {
			.size = pixels,
			.usage = buffer_usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.usage_type = BFM_READ_ONLY
		}
	};

	_bf_debug = Buffer4f {
		context,
		BFM_Settings {
			.size = pixels,
			.usage = buffer_usage,
			.usage_type = BFM_READ_ONLY
		}
	};

	// BFM_Settings for write buffers
	BFM_Settings bfm_wo_settings {
		.size = 1024,
		.usage = buffer_usage,
		.usage_type = BFM_WRITE_ONLY
	};

	// Write onlt buffers
	_bf_world = BufferManager <uint8_t> {context, bfm_wo_settings};
	_bf_objects = Buffer4f {context, bfm_wo_settings};
	_bf_lights = Buffer4f {context, bfm_wo_settings};
	_bf_materials = Buffer4f {context, bfm_wo_settings};
	_bf_vertices = Buffer4f {context, bfm_wo_settings};
	_bf_transforms = Buffer4m {context, bfm_wo_settings};

	_bf_textures = Buffer4f {context, bfm_wo_settings};
	_bf_texture_info = Buffer4u {context, bfm_wo_settings};

	// TODO: initialization function (preload)

	// Load sky texture
	Texture sky = load_image_texture("resources/sky.jpg");

	// Upload sky texture
	raytracing::TextureUpdate tu {
		.textures = &_bf_textures,
		.texture_info = &_bf_texture_info,
	};

	tu.reset();
	tu.write(sky);
	tu.upload();
}

// Dump debug data to file
void RTApp::dump_debug_data(Vulkan *vk)
{
	// Open file
	std::ofstream file("debug.log");

	file << "=== Debug data ===" << std::endl;

	// Wait for queue to finish
	vkQueueWaitIdle(context.device.graphics_queue);

	// Extract data from debug buffer
	const aligned_vec4 *data = _bf_debug.data();

	// Dump data
	for (size_t i = 0; i < 800 * 600; i++) {
		// Cast to quads of ints
		glm::vec4 vec = data[i].data;
		int *ptr1 = (int *) &vec.x;
		int *ptr2 = (int *) &vec.y;
		int *ptr3 = (int *) &vec.z;
		int *ptr4 = (int *) &vec.w;

		file << vec << " --> " << *ptr1 << ", "
			<< *ptr2 << ", " << *ptr3 << ", " << *ptr4 << std::endl;
	}
}

// Create ImGui profiler tree
void RTApp::make_profiler_tree(const kobra::Profiler::Frame &frame, float parent)
{
	// Show tree
	if (ImGui::TreeNode(frame.name.c_str())) {
		ImGui::Text("time:   %10.3f ms", frame.time);

		if (parent > 0) {
			float percent = frame.time / parent;
			ImGui::Text("parent: %10.3f%%", percent * 100.0f);
		}

		for (auto &child : frame.children)
			make_profiler_tree(child, frame.time);
		ImGui::TreePop();
	}
}

// Create ImGui render
void RTApp::make_imgui(size_t image_index)
{
	// Fill out imgui command buffer and render pass
	context.vk->begin_command_buffer(imgui_ctx.command_buffer);

	// Begin the render pass
	context.vk->begin_render_pass(
		imgui_ctx.command_buffer,
		swapchain.framebuffers[image_index],
		imgui_ctx.render_pass,
		swapchain.extent,
		0, nullptr
	);

		// ImGui new frame
		// TODO: method
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Show render monitor
		ImGui::Begin("Render Monitor");
		{
			ImGui::Text("fps: %.1f", ImGui::GetIO().Framerate);
			ImGui::Checkbox("BVH Debugging", &world.options.debug_bvh);
			ImGui::InputInt("Descretize (grey)", &world.options.discretize);

			if (ImGui::Button("Capture Debug Data"))
				dump_debug_data(context.vk);
		}
		ImGui::End();

		// Statistics
		ImGui::Begin("Statistics");
		{
			ImGui::Text("Objects: %u", gworld.objects);
			ImGui::Text("Primitives: %u", gworld.primitives);
			ImGui::Text("Lights:  %u", gworld.lights);

			if (ImGui::TreeNode("BVH")) {
				ImGui::Text("# Nodes: %lu", bvh.size);
				ImGui::Text("# Primitives: %lu", bvh.primitives);
				ImGui::TreePop();
			}

			// Buffer sizes in MB
			auto to_mb = [](size_t size) {
				return size / (1024.0f * 1024.0f);
			};

			// TODO: function
			if (ImGui::TreeNode("Buffer sizes")) {
				ImGui::Text("Pixel buffer:     %5.2f MB", to_mb(_bf_pixels.bytes()));

				ImGui::Separator();
				ImGui::Text("World buffer:     %5.2f MB", to_mb(_bf_world.bytes()));
				ImGui::Text("Objects buffer:   %5.2f MB", to_mb(_bf_objects.bytes()));
				ImGui::Text("Lights buffer:    %5.2f MB", to_mb(_bf_lights.bytes()));
				ImGui::Text("Materials buffer: %5.2f MB", to_mb(_bf_materials.bytes()));
				ImGui::Text("Vertex buffer:    %5.2f MB", to_mb(_bf_vertices.bytes()));
				ImGui::Text("Transform buffer: %5.2f MB", to_mb(_bf_transforms.bytes()));

				ImGui::Separator();
				ImGui::Text("Textures buffer:   %5.2f MB", to_mb(_bf_textures.bytes()));
				ImGui::Text("TexInfos buffer:   %5.2f MB", to_mb(_bf_texture_info.bytes()));

				ImGui::Separator();
				ImGui::Text("BVH buffer:       %5.2f MB", to_mb(bvh.buffer.size));
				ImGui::Text("Debug buffer:     %5.2f MB", to_mb(_bf_debug.bytes()));
				ImGui::TreePop();
			}
		}
		ImGui::End();

		// Image/video capture
		ImGui::Begin("Capture");
		{
			if (ImGui::Button("Capture image")) {
				vkQueueWaitIdle(context.device.graphics_queue);
				Image image {.width = 800, .height = 600};
				Capture::snapshot(_bf_pixels, image);
				image.write("capture.png");
			}

			// TODO: will need to set the fps of the phsyics simulation
			std::string button = "Start capturing";
			if (capturing)
				button = "Stop capturing";

			if (ImGui::Button(button.c_str())) {
				if (!capturing) {
					capturing = true;
					capture_timer.start();
					// capture.start("capture.avi", 800, 600);

					Capture::Format fmt {
						.bitrate = 1000000,
						.width = 800,
						.height = 600,
						.framerate = 60,
						.gop = 60
					};

					capture.start("capture.avi", fmt);
				} else {
					capturing = false;
				}
			} else {
				if (capturing) {
					ImGui::Text(
						"Capture (real time): %5.3f s",
						capture_timer.elapsed_start() / 1000000.0f
					);

					ImGui::Text(
						"Capture (video time): %5.3f s",
						capture.time()
					);

					capture.write(_bf_pixels);
				} else {
					capture.flush();
				}
			}
		}
		ImGui::End();

		/* if (profiler.size() > 0) {
			auto frame = profiler.pop();

			ImGui::Begin("Profiler");
			make_profiler_tree(frame);
			ImGui::End();
		} */

		ImGui::EndFrame();
		ImGui::Render();

		// Render ImGui
		ImGui_ImplVulkan_RenderDrawData(
			ImGui::GetDrawData(),
			imgui_ctx.command_buffer
		);

	// End the render pass
	context.vk->end_render_pass(imgui_ctx.command_buffer);

	// End the command buffer
	context.vk->end_command_buffer(imgui_ctx.command_buffer);
}

// Constructor
RTApp::RTApp(Vulkan *vk)
		: kobra::App({
			.ctx = vk,
			.width = 800,
			.height = 600,
			.name = "Mercury - Sample Scene",
		})
{
	// GLFW callbacks
	glfwSetKeyCallback(surface.window, key_callback);
	glfwSetInputMode(surface.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(surface.window, mouse_callback);

	// Initialize ImGui for this application
	imgui_ctx = context.vk->init_imgui_glfw(context.phdev, context.device, surface, swapchain);

	// Create render pass
	render_pass = context.vk->make_render_pass(
		context.device,
		swapchain,
		VK_ATTACHMENT_LOAD_OP_LOAD,
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
	// TODO: context method
	descriptor_set_layout = context.vk->make_descriptor_set_layout(context.device, compute_dsl_bindings);
	preproc_dsl = context.vk->make_descriptor_set_layout(context.device, preproc_dsl_bindings);

	// Create descriptor sets
	descriptor_set = context.vk->make_descriptor_set(
		context.device,
		descriptor_pool,
		descriptor_set_layout
	);

	preproc_ds = context.vk->make_descriptor_set(
		context.device,
		descriptor_pool,
		preproc_dsl
	);

	// Load compute shader
	compute_shader = context.vk->make_shader(context.device, "shaders/bin/generic/pixel.spv");
	pp_vert_shader = context.vk->make_shader(context.device, "shaders/bin/generic/pp_vert.spv");
	pp_frag_shader = context.vk->make_shader(context.device, "shaders/bin/generic/pp_frag.spv");

	// Create sync objects
	// TODO: use max frames in flight
	images_in_flight.resize(swapchain.images.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < 2; i++) {
		in_flight_fences.push_back(context.vk->make_fence(context.device, VK_FENCE_CREATE_SIGNALED_BIT));
		smph_image_available.push_back(context.vk->make_semaphore(context.device));
		smph_render_finished.push_back(context.vk->make_semaphore(context.device));
	}

	// Create the buffers
	allocate_buffers();

	// Create BVH builder
	bvh = kobra::BVH(context.vk, context.phdev, context.device, world);
	Logger::ok() << "BVH: " << bvh.size << " nodes, " << bvh.primitives << " primitives" << std::endl;

	kobra::BVHNode *root = bvh.nodes[0];
	Logger::warn() << "Checking BVH: should traverse through " << root->size()
		<< " nodes if proper." << std::endl;

	Buffer bvh_buf;
	root->write(bvh_buf);

	auto leaf = [&](int node) {
		return (bvh_buf[node].data.x == 0x1);
	};

	auto hit = [&](int node) {
		return *reinterpret_cast <int32_t *> (&bvh_buf[node].data.z);
	};

	int node = 0;
	int count = 0;
	while (node != -1) {
		count++;

		// Always go to hit
		node = hit(node);
	}

	Logger::warn() << "Traversed through " << count << " nodes." << std::endl;

	if (count != root->size())
		Logger::error() << "BVH traversal failed!" << std::endl;
}

// Update the world
void RTApp::update_world() {
	static float time = 0.0f;

	// Update light position
	float amplitude = 7.0f;
	glm::vec3 position {
		amplitude * sin(time), 7.0f,
		amplitude * cos(time)
	};

	world.objects[0]->transform().position = position;
	world.lights[0]->transform.position = position;

	// Map buffers
	if (map_buffers(context.vk)) {
		update_descriptor_set();
		update_command_buffers();
	}

	bvh.update(world);

	/* Print contents of bvh buffer
	Logger::ok() << "[main] BVH buffer contents\n";
	for (size_t i = 0; i < bvh.dump.size(); i += 3) {
		glm::ivec4 dump = *reinterpret_cast <glm::ivec4 *> (&bvh.dump[i].data);
		std::cou t << "\t" << i << ": " << bvh.dump[i] << " --> " << dump << std::endl;
	} */

	// Update time
	time += frame_time;
}

// Present the frame
void RTApp::present()
{
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
	profiler.frame("Acquire image");
	if (images_in_flight[image_index] != VK_NULL_HANDLE) {
		vkWaitForFences(
			context.vk_device(), 1,
			&images_in_flight[image_index],
			VK_TRUE, UINT64_MAX
		);
	}
	profiler.end();

	// Mark the image as in use by this frame
	images_in_flight[image_index] = in_flight_fences[frame_index];

	// Render imgui
	make_imgui(image_index);

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
		imgui_ctx.semaphore
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

	profiler.frame("Submit RT command buffer");
	{
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
	}
	profiler.end();

	profiler.frame("ImGui command buffer");
	{
		// Wait for the first command buffer to finish
		// TODO: use wait semaphores
		vkQueueWaitIdle(context.device.graphics_queue);

		// Submit ImGui command buffer
		submit_info.waitSemaphoreCount = 0;
		submit_info.pSignalSemaphores = &imgui_ctx.semaphore;
		submit_info.pCommandBuffers = &imgui_ctx.command_buffer;

		// Submit the command buffer
		// TODO: Vulkan method
		vkResetFences(context.vk_device(), 1, &imgui_ctx.fence);
		result = vkQueueSubmit(
			context.device.graphics_queue, 1, &submit_info,
			imgui_ctx.fence
		);

		if (result != VK_SUCCESS) {
			Logger::error("[main] Failed to submit draw ImGui command buffer!");
			throw (-1);
		}
	}
	profiler.end();

	profiler.frame("vkQueuePresentKHR");
	{
		// Present the image to the swap chain
		VkSwapchainKHR swchs[] = {swapchain.swch};

		VkPresentInfoKHR present_info {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 2,
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
	profiler.end();
}

void RTApp::frame()
{
	// Start profiling a new frame
	profiler.frame("Frame");

	// Update world
	profiler.frame("Update world");
		update_world();
	profiler.end();

	// Present the frame
	profiler.frame("Present frame");
		present();
	profiler.end();

	// End profiling the frame and display it
	profiler.end();
}

void RTApp::update_command_buffers() {
	// Set command buffers
	auto ftn = [this](const Vulkan *ctx, size_t i) {
		// TODO: maker should be a virtual function
		this->maker(ctx, i);
	};

	context.vk->set_command_buffers(
		context.device,
		swapchain, command_pool,
		command_buffers, ftn
	);
}

// TODO: make some cleaner method
void RTApp::update_descriptor_set()
{
	// TODO: fix up BVH situation
	VkDescriptorBufferInfo bvh_info {
		.buffer = bvh.buffer.buffer,
		.offset = 0,
		.range = bvh.buffer.size
	};

	VkDescriptorBufferInfo stack_info {
		.buffer = bvh.stack.buffer,
		.offset = 0,
		.range = bvh.stack.size
	};

	VkWriteDescriptorSet bvh_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = 5,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &bvh_info
	};

	VkWriteDescriptorSet stack_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = 6,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &stack_info
	};

	VkWriteDescriptorSet writes[] = {
		bvh_write,
		stack_write
	};

	vkUpdateDescriptorSets(
		context.device.device, 2,
		&writes[0],
		0, nullptr
	);

	// Make indiviual descriptor set udpates
	_bf_pixels.bind(descriptor_set, 0);
	_bf_world.bind(descriptor_set, 1);
	_bf_objects.bind(descriptor_set, 2);
	_bf_lights.bind(descriptor_set, 3);
	_bf_materials.bind(descriptor_set, 4);
	_bf_debug.bind(descriptor_set, 7);
	_bf_vertices.bind(descriptor_set, 8);
	_bf_transforms.bind(descriptor_set, 9);

	// Texture module
	_bf_textures.bind(descriptor_set, 10);
	_bf_texture_info.bind(descriptor_set, 11);

	///////////////////////////
	// Preprocessing shaders //
	///////////////////////////

	_bf_pixels.bind(preproc_ds, 0);
	_bf_world.bind(preproc_ds, 1);
}
