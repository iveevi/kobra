// Standard headers
#include <cstring>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>
// #include <vulkan/vulkan_core.h>

// Local headers
#include "global.hpp"
#include "include/buffer_manager.hpp"
#include "include/world.hpp"

using namespace mercury;

// List of objet materials
Material materials[] {
	{.albedo = glm::vec3 {0.1f, 0.5f, 0.2f}},
	{
		.albedo = glm::vec3 {0.9f, 0.5f, 0.2f},
		.specular = 0.5f,
		.reflectance = 0.2f,
	},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.specular = 1.0,
		.reflectance = 0.7
	},
	{.albedo = glm::vec3 {0.5f, 0.1f, 0.6f}},
	{.albedo = glm::vec3 {0.6f, 0.5f, 0.3f}},
	{.albedo = glm::vec3 {1.0f, 0.5f, 1.0f}},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.shading = SHADING_TYPE_LIGHT
	}
};

// List of object transforms
Transform transforms[] {
	glm::vec3 {-1.0f, 0.0f, 4.0f},
	glm::vec3 {0.5f, 5.0f, 3.5f},
	glm::vec3 {6.0f, -2.0f, 5.0f},
	glm::vec3 {6.0f, 3.0f, 11.5f},
	glm::vec3 {6.0f, 3.0f, -2.0f},
	glm::vec3 {0.0f, 0.0f, 0.0f}
};

// Print aligned_vec4
// TODO: common header
inline std::ostream& operator<<(std::ostream& os, const glm::vec4 &v)
{
	return (os << "(" << v.x << ", " << v.y
		<< ", " << v.z << ", " << v.w << ")");
}

inline std::ostream &operator<<(std::ostream &os, const aligned_vec4 &v)
{
	return (os << v.data);
}

// Print BoundingBox
inline std::ostream &operator<<(std::ostream &os, const mercury::BoundingBox &b)
{
	return (os << "(" << b.min << " --> " << b.max << ")");
}

World world {
	// Camera
	Camera {
		Transform {
			glm::vec3(0.0f, 0.0f, -4.0f)
		},
	 
		Tunings {
			45.0f, 800, 600
		}
	},

	// Primitives
	// TODO: later read from file
	std::vector <World::PrimitivePtr> {
		World::PrimitivePtr(new Sphere(0.25f, transforms[0], materials[6])),
		World::PrimitivePtr(new Sphere(1.0f, transforms[0], materials[0])),
		World::PrimitivePtr(new Sphere(3.0f, transforms[1], materials[1])),
		World::PrimitivePtr(new Sphere(6.0f, transforms[2], materials[2])),
		World::PrimitivePtr(new Sphere(2.0f, transforms[3], materials[3])),
		World::PrimitivePtr(new Sphere(2.0f, transforms[4], materials[4])),

		// Cube mesh
		World::PrimitivePtr(new mercury::Mesh <mercury::VERTEX_TYPE_POSITION> (
			{
				glm::vec3(0.0f, 6.0f, -1.5f),
				glm::vec3(1.0f, 6.0f, -1.5f),
				glm::vec3(1.0f, 7.0f, -1.5f),
				glm::vec3(0.0f, 7.0f, -1.5),
				glm::vec3(0.0f, 6.0f, 0.5f),
				glm::vec3(1.0f, 6.0f, 0.5f),
				glm::vec3(1.0f, 7.0f, 0.5f),
				glm::vec3(0.0f, 7.0f, 0.5f)
			},
			{
				0, 1, 2,	0, 2, 3,
				4, 5, 6,	4, 6, 7,
				0, 4, 7,	0, 7, 3,
				1, 5, 6,	1, 6, 2,
				0, 1, 4,	1, 4, 5,
				2, 6, 7,	2, 7, 3
			},
			materials[1]
		)),
	},

	// Lights
	std::vector <World::LightPtr> {
		// TODO: objects with emmision
		World::LightPtr(new PointLight(transforms[0], 0.0f))
	}
};

// TODO: put the following functions into a alloc.cpp file

// Minimum sizes
#define INITIAL_OBJECTS		128UL
#define INITIAL_LIGHTS		128UL
#define INITIAL_MATERIALS	128UL

// Sizes of objects and lights
// are assumed to be the maximum
static const size_t MAX_OBJECT_SIZE = sizeof(Triangle);
static const size_t MAX_LIGHT_SIZE = sizeof(PointLight);

// App
class MercuryApplication : public mercury::App {
	// TODO: some of these member should be moved back to App
	VkRenderPass			render_pass;
	VkCommandPool			command_pool;

	std::vector <VkCommandBuffer>	command_buffers;

	VkDescriptorPool		descriptor_pool;
	VkDescriptorSetLayout		descriptor_set_layout;
	VkDescriptorSet			descriptor_set;

	VkShaderModule			compute_shader;

	// Sync objects
	std::vector <VkFence>		in_flight_fences;
	std::vector <VkFence>		images_in_flight;

	std::vector <VkSemaphore>	smph_image_available;
	std::vector <VkSemaphore>	smph_render_finished;

	// Profiler
	Profiler		profiler;

	// Fill out command buffer
	// TODO: do we need the vk parameter?
	void maker(const Vulkan *vk, size_t i) {
		// Get image at current index
		VkImage image = swapchain.images[i];

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
			command_buffers[i],
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &image_memory_barrier
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
				swapchain.extent.width,
				swapchain.extent.height,
				1
			}
		};

		// Copy buffer to image
		vkCmdCopyBufferToImage(
			command_buffers[i],
			_bf_pixels.vk_buffer(),
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&buffer_copy_region
		);
		
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

			// Bind pipeline
			vkCmdBindPipeline(
				command_buffers[i],
				VK_PIPELINE_BIND_POINT_COMPUTE,
				pipeline
			);

			// Bind buffer
			vkCmdBindDescriptorSets(
				command_buffers[i],
				VK_PIPELINE_BIND_POINT_COMPUTE,
				pipeline_layout,
				0, 1, &descriptor_set,
				0, nullptr
			);

		// TODO: use the methods
		vkCmdEndRenderPass(command_buffers[i]);
			
		// Dispatch
		vkCmdDispatch(
			command_buffers[i],
			swapchain.extent.width,
			swapchain.extent.height, 1
		);
	}
	
	// GPU buffers
	// Vulkan::Buffer	objects_buffer;
	Vulkan::Buffer	lights_buffer;
	Vulkan::Buffer	materials_buffer;

	BufferManager <uint>	_bf_pixels;
	BufferManager <uint8_t>	_bf_world;
	Buffer4f		_bf_objects;
	Buffer4f		_bf_lights;
	Buffer4f		_bf_materials;
	Buffer4f		_bf_debug;

	// BVH resources
	mercury::BVH bvh;

	////////////////////
	// Buffer methods //
	////////////////////

	// Copy buffer helper
	GPUWorld gworld;

	// Map all the buffers
	// TODO: deal with resizing buffers
	bool map_buffers(Vulkan *vk) {
		// TODO: remove this
		static const VkBufferUsageFlags buffer_usage =
			VK_BUFFER_USAGE_TRANSFER_DST_BIT
			| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		
		// Reset pushback indices
		_bf_objects.reset_push_back();
		
		// Resizing and remaking buffers
		int resized = 0;

		// World update
		WorldUpdate wu {
			.bf_objs = &_bf_objects
		};

		// Generate world data and write to buffers
		// Indices indices;
		world.write_objects(wu);
		world.write_lights(wu);
		// TODO: simple write method

		// Calculate size of world buffer
		size_t world_size = sizeof(GPUWorld) + wu.indices.size() * sizeof(uint);

		// Copy world and indices
		gworld = world.dump();
		gworld.objects = wu.bf_objs->push_size();	// TODO: this is not the correct size

		if (_bf_world.size() < world_size) {
			Logger::notify() << "Resizing world buffer" << std::endl;
			resized += _bf_world.resize(world_size);
		}
		
		_bf_world.write(
			(const uint8_t *) &gworld,
			sizeof(GPUWorld)
		);

		_bf_world.write(
			(const uint8_t *) wu.indices.data(),
			4 * wu.indices.size(),
			sizeof(GPUWorld)
		);

		if (_bf_objects.sync_size())
			Logger::notify("Resizing objects buffer");

		// Lights
		if (sizeof(aligned_vec4) * wu.lights.size() > lights_buffer.size) {
			// Resize vulkan buffer
			vk->destroy_buffer(context.device, lights_buffer);
			vk->make_buffer(context.phdev, context.device, lights_buffer, wu.lights.size() * sizeof(aligned_vec4), buffer_usage);
			resized++;
		}

		// Map buffers
		_bf_world.flush();
		_bf_objects.flush();
		vk->map_buffer(context.device, &materials_buffer, wu.materials.data(), sizeof(Material) * wu.materials.size());
		vk->map_buffer(context.device, &lights_buffer, wu.lights.data(), sizeof(aligned_vec4) * wu.lights.size());

		// Return pointer to the buffer
		return (resized > 0);
	}

	// Allocate buffers
	void allocate_buffers() {
		// Allocate buffers
		size_t pixel_size = 4 * 800 * 600;
		size_t objects_size = MAX_OBJECT_SIZE * INITIAL_OBJECTS;
		size_t lights_size = MAX_LIGHT_SIZE * std::max(world.lights.size(), INITIAL_LIGHTS);
		size_t materials_size = sizeof(Material) * INITIAL_MATERIALS;

		static const VkBufferUsageFlags buffer_usage =
			VK_BUFFER_USAGE_TRANSFER_DST_BIT
			| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		// Create buffers
		// context.vk->make_buffer(physical_device, device, pixel_buffer, pixel_size, buffer_usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

		// context.vk->make_buffer(context.phdev, context.device, world_buffer, world_size, buffer_usage);
		// context.vk->make_buffer(context.phdev, context.device, objects_buffer, objects_size, buffer_usage);
		context.vk->make_buffer(context.phdev, context.device, lights_buffer, lights_size, buffer_usage);
		context.vk->make_buffer(context.phdev, context.device, materials_buffer, materials_size, buffer_usage);

		// Debug buffer (one vec4 per pixel)
		// context.vk->make_buffer(physical_device, device, debug_buffer, 800 * 600 * sizeof(aligned_vec4), buffer_usage);

		// TODO: clean, lots of duplicate code
		_bf_pixels = BufferManager <uint> {
			context,
			BFM_Settings {
				.size = pixel_size,
				.usage = buffer_usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				.usage_type = BFM_READ_ONLY
			}
		};

		_bf_world = BufferManager <uint8_t> {
			context,
			BFM_Settings {
				// Allocate dummy amount (1 KB)
				.size = 1024,
				.usage = buffer_usage,
				.usage_type = BFM_WRITE_ONLY
			}
		};

		_bf_objects = Buffer4f {
			context,
			BFM_Settings {
				.size = 1024,
				.usage = buffer_usage,
				.usage_type = BFM_WRITE_ONLY
			}
		};

		_bf_lights = Buffer4f {
			context,
			BFM_Settings {
				.size = 1024,
				.usage = buffer_usage,
				.usage_type = BFM_WRITE_ONLY
			}
		};

		_bf_materials = Buffer4f {
			context,
			BFM_Settings {
				.size = 1024,
				.usage = buffer_usage,
				.usage_type = BFM_WRITE_ONLY
			}
		};

		_bf_debug = Buffer4f {
			context,
			BFM_Settings {
				.size = 800 * 600,
				.usage = buffer_usage,
				.usage_type = BFM_READ_ONLY
			}
		};
	}

	// ImGui context and methods
	// TODO: the context should not have any sync objects
	Vulkan::ImGuiContext imgui_ctx;

	// TODO: wrap inside a struct
	bool capturing = false;
	mercury::Timer capture_timer;
	mercury::Capture capture;

	// Dump debug data to file
	void dump_debug_data(Vulkan *vk) {
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
	void make_profiler_tree(const mercury::Profiler::Frame &frame, float parent = -1.0) {
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
	void make_imgui(size_t image_index) {
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

				if (ImGui::TreeNode("Buffer sizes")) {
					ImGui::Text("Pixel buffer:     %5.2f MB", to_mb(_bf_pixels.size()));
					ImGui::Text("World buffer:     %5.2f MB", to_mb(_bf_world.size()));
					ImGui::Text("Objects buffer:   %5.2f MB", to_mb(_bf_objects.size()));
					ImGui::Text("Lights buffer:    %5.2f MB", to_mb(lights_buffer.size));
					ImGui::Text("Materials buffer: %5.2f MB", to_mb(materials_buffer.size));
					ImGui::Text("BVH buffer:       %5.2f MB", to_mb(bvh.buffer.size));
					ImGui::Text("Debug buffer:     %5.2f MB", to_mb(_bf_debug.size()));
					ImGui::TreePop();
				}
			}
			ImGui::End();

			// Image/video capture
			ImGui::Begin("Capture");
			{
				if (ImGui::Button("Capture image")) {
					vkQueueWaitIdle(context.device.graphics_queue);
					mercury::capture::write(
						context.vk, context.device,
						_bf_pixels.buffer(),
						800, 600,
						"capture.png"
					);
				}

				// TODO: will need to set the fps of the phsyics simulation
				std::string button = "Start capturing";
				if (capturing)
					button = "Stop capturing";
				
				if (ImGui::Button(button.c_str())) {
					if (!capturing) {
						capturing = true;
						capture_timer.start();
						capture.start("capture.avi", 800, 600);
					} else {
						capturing = false;
					}
				} else {
					if (capturing) {
						ImGui::Text(
							"Capture: %5.3f s",
							capture_timer.elapsed_start() / 1000000.0f
						);

						capture.write(
							context.vk, context.device,
							_bf_pixels.buffer(),
							800, 600
						);
					} else {
						capture.flush();
					}
				}
			}
			ImGui::End();
			
			if (profiler.size() > 0) {
				auto frame = profiler.pop();

				ImGui::Begin("Profiler");
				make_profiler_tree(frame);
				ImGui::End();
			}

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
public:
	// Constructor
	MercuryApplication(Vulkan *vk) : mercury::App({
		.ctx = vk,
		.width = 800,
		.height = 600,
		.name = "Mercury - Sample Scene",
	}) {
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
		descriptor_set_layout = context.vk->make_descriptor_set_layout(context.device, dsl_bindings);

		// Create descriptor set
		descriptor_set = context.vk->make_descriptor_set(
			context.device,
			descriptor_pool,
			descriptor_set_layout
		);

		// Load compute shader
		compute_shader = context.vk->make_shader(context.device, "shaders/pixel.spv");

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
		bvh = mercury::BVH(context.vk, context.phdev, context.device, world);
		Logger::ok() << "BVH: " << bvh.size << " nodes, " << bvh.primitives << " primitives" << std::endl;

		mercury::BVHNode *root = bvh.nodes[0];
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
	void update_world() {
		static float time = 0.0f;

		// Update light position
		float amplitude = 7.0f;
		glm::vec3 position {
			amplitude * sin(time), 7.0f,
			amplitude * cos(time)
		};

		world.objects[0]->transform.position = position;
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

		// Submit the command buffer
		vkResetFences(context.device.device, 1, &in_flight_fences[frame_index]);
	
		profiler.frame("Queue submit");
		result = vkQueueSubmit(
			context.device.graphics_queue, 1, &submit_info,
			in_flight_fences[frame_index]
		);
		profiler.end();

		if (result != VK_SUCCESS) {
			Logger::error("[main] Failed to submit draw command buffer!");
			throw (-1);
		}

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

		profiler.frame("vkQueuePresentKHR");
		result = vkQueuePresentKHR(context.device.present_queue, &present_info);
		profiler.end();
		
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

	void frame() override {
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

	void update_command_buffers() {
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
	void update_descriptor_set() {
		/* VkDescriptorBufferInfo wb_info {
			.buffer = world_buffer.buffer,
			.offset = 0,
			.range = world_buffer.size
		}; */
		
		/* VkDescriptorBufferInfo ob_info {
			.buffer = objects_buffer.buffer,
			.offset = 0,
			.range = objects_buffer.size
		}; */

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
		
		/* VkWriteDescriptorSet wb_write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 1,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &wb_info
		}; */
		
		/* VkWriteDescriptorSet ob_write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 2,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &ob_info
		}; */

		VkWriteDescriptorSet lb_write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 3,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &lb_info
		};

		VkWriteDescriptorSet mt_write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 4,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &mt_info
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
			// wb_write,
			// ob_write,
			lb_write,
			mt_write,
			bvh_write,
			stack_write
		};

		vkUpdateDescriptorSets(
			context.device.device, 4,
			&writes[0],
			0, nullptr
		);

		_bf_pixels.update_descriptor_set(descriptor_set, 0);
		_bf_world.update_descriptor_set(descriptor_set, 1);
		_bf_objects.update_descriptor_set(descriptor_set, 2);
		_bf_debug.update_descriptor_set(descriptor_set, 7);
	}

	// Desctiptor set layout bindings
	static const std::vector <VkDescriptorSetLayoutBinding> dsl_bindings;
};

// TODO: cache in constructor or something...
const std::vector <VkDescriptorSetLayoutBinding> MercuryApplication::dsl_bindings = {
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
	}
};

// Profiler application
class ProfilerApplication : public mercury::App {

};

int main()
{
	// Redirect logger to file
	// Logger::switch_file("mercury.log");

	mercury::Model <mercury::VERTEX_TYPE_POSITION> model("resources/benchmark/suzanne.obj");
	model[0].material = materials[1];

	world.objects.push_back(std::shared_ptr <mercury::Model <mercury::VERTEX_TYPE_POSITION>> (
		new mercury::Model <mercury::VERTEX_TYPE_POSITION> (model)
	));

	Logger::ok() << "[main] Loaded model with "
		<< model.mesh_count() << " meshe(s), "
		<< model[0].vertex_count() << " vertices, "
		<< model[0].triangle_count() << " triangles" << std::endl;

	// Save world into scene
	mercury::Scene scene("default_world", world);
	scene.save("resources/default_world.hg");

	// Initialize Vulkan
	Vulkan *vulkan = new Vulkan();
	vulkan->init_imgui();

	// Create sample scene
	MercuryApplication app(vulkan);

	app.update_descriptor_set();
	app.update_command_buffers();
	app.run();

	delete vulkan;
}