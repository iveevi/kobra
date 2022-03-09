#ifndef PROFILER_APPLICATION_H_
#define PROFILER_APPLICATION_H_

#include "global.hpp"
#include "include/gui/gui.hpp"
#include "include/gui/text.hpp"
#include "include/gui/area.hpp"
#include "include/gui/button.hpp"
#include "include/gui/layer.hpp"

using namespace mercury;

// Profiler application
class ProfilerApplication : public mercury::App {
	// Vulkan structures
	VkRenderPass			render_pass;
	VkCommandPool			command_pool;
	std::vector <VkCommandBuffer>	command_buffers;
	VkDescriptorPool		descriptor_pool;

	// Sync objects
	std::vector <VkFence>		in_flight_fences;
	std::vector <VkFence>		images_in_flight;

	std::vector <VkSemaphore>	smph_image_available;
	std::vector <VkSemaphore>	smph_render_finished;

	// Reference to the profiler
	Profiler *			profiler;

	// GUI layers
	gui::Layer			layer;

	// Objects to save
	gui::Text *			t1;
public:
	// TODO: base app class that initializes vulkun structures and does the
	// presenting
	ProfilerApplication(Vulkan *vk, Profiler *p) : App({
		.ctx = vk,
		.width = 800,
		.height = 600,
		.name = "Mercury Profiler"
	}), profiler(p) {
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
		// TODO: context method
		command_pool = context.vk->make_command_pool(
			context.phdev,
			surface,
			context.device,
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
		);

		// Create descriptor pool
		descriptor_pool = context.vk->make_descriptor_pool(context.device);

		// Copy to window context
		window.command_pool = command_pool;
		window.descriptor_pool = descriptor_pool;

		// Create sync objects
		// TODO: use max frames in flight
		images_in_flight.resize(swapchain.images.size(), VK_NULL_HANDLE);
		for (size_t i = 0; i < 2; i++) {
			in_flight_fences.push_back(context.vk->make_fence(context.device, VK_FENCE_CREATE_SIGNALED_BIT));
			smph_image_available.push_back(context.vk->make_semaphore(context.device));
			smph_render_finished.push_back(context.vk->make_semaphore(context.device));
		}

		// TODO: context method
		context.vk->make_command_buffers(
			context.device,
			command_pool,
			command_buffers,
			swapchain.images.size()
		);

		// Create GUI layer
		gui::Button *button = new gui::Button(window,
			{
				400, 500, 100, 100,
				GLFW_MOUSE_BUTTON_LEFT,
				{0.8, 1.0, 0.8},
				{0.6, 1.0, 0.6},
				{0.4, 1.0, 0.4}
			}
		);

		// Initialize GUI layers
		layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_CLEAR);
		layer.load_font("default", "resources/fonts/noto_sans.ttf");

		std::vector <gui::_element *> elements {
			button
		};

		layer.add(elements);

		make_window();
	}

	// Element that contains the profiler info
	gui::Rect *content = nullptr;

	void make_window() {
		int x = 400;
		int y = 200;

		gui::Text *t = layer.text_render("default")->text(
			"My window", {x, y}, {1, 1, 1, 1}
		);

		glm::vec4 bounds = t->bounds;
		bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};

		glm::vec3 color = {0.0, 0.5, 1.0};

		gui::Rect *tborder = new gui::Rect(bounds, color * 0.5f);

		glm::vec4 cbounds = bounds + 0.2f * glm::vec4 {0, 1, 0, 1};
		gui::Rect *cborder = new gui::Rect(cbounds, color * 0.5f);

		bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};
		gui::Rect *wborder = new gui::Rect(bounds, color);

		tborder->children.push_back(gui::Element(t));
		wborder->children.push_back(gui::Element(tborder));
		wborder->children.push_back(gui::Element(cborder));

		layer.add(wborder);

		content = cborder;
	}

	// Update the profiler
	void update_profiler(const Profiler::Frame &frame, gui::Rect *element, float parent = -1.0f) {
		glm::vec3 color = {0.0, 0.5, 1.0};
		glm::vec2 tpos = element->position();
		float scale = 0.4f;

		// Only add if not already added
		if (element->children.size() >= 3) {
			std::cout << "Frame \"" << frame.name << "\", position = " << tpos.x << ", " << tpos.y << std::endl;
			std::cout << "\telement position = " << element->position().x << ", " << element->position().y << std::endl;

			// Time and parent have been expanded
			gui::Text *f = reinterpret_cast <gui::Text *> (element->children[0].get());
			gui::Text *t = reinterpret_cast <gui::Text *> (element->children[1].get());
			gui::Text *p = reinterpret_cast <gui::Text *> (element->children[2].get());
			
			// Update positions
			f->pos = tpos;
			t->pos = tpos + glm::vec2 {0, 0.01};
			p->pos = tpos + glm::vec2 {0, 0.02};

			f->refresh();
			t->refresh();
			p->refresh();

			std::cout << "\t\tf->bounds = " << f->bounds.x << ", " << f->bounds.y << ", " << f->bounds.z << ", " << f->bounds.w << std::endl;
			std::cout << "\t\tt->bounds = " << t->bounds.x << ", " << t->bounds.y << ", " << t->bounds.z << ", " << t->bounds.w << std::endl;
			std::cout << "\t\tp->bounds = " << p->bounds.x << ", " << p->bounds.y << ", " << p->bounds.z << ", " << p->bounds.w << std::endl;

			t->str = "time: " + std::to_string(frame.time/1000) + " us";

			std::string parent_str = "parent: ";
			if (parent > 0) {
				float ratio = 100 * (frame.time / parent);
				parent_str += std::to_string(ratio) + "%";
			} else {
				parent_str += "-NA-";
			}

			p->str = parent_str;

			// Set bounds of element rect
			glm::vec4 bounds = gui::get_bounding_box({f, t, p});
			bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};

			std::cout << "\tinitial bounds position = " << bounds.x
				<< ", " << bounds.y
				<< ", " << bounds.z
				<< ", " << bounds.w << std::endl;

			/* Go through each child box
			size_t n = element->children.size();

			float y = bounds.w + 0.01;
			for (size_t i = 3; i < n; i++) {
				gui::Rect *child = reinterpret_cast <gui::Rect *>
					(element->children[i].get());
				glm::vec4 cbounds = {
					bounds.x, y,
					0, 0
				};
				child->set_bounds(cbounds);

				// Update child
				update_profiler(frame.children[i-3], child, frame.time);
				y = child->bounding_box().w + 0.01;
			}

			std::vector <gui::Element> children = element->children;
			// children.erase(children.begin(), children.begin() + 3);
			bounds = gui::get_bounding_box(children);
			std::cout << "\tupdated bounds position = " << bounds.x << ", " << bounds.y << std::endl;
			// element->set_bounds(bounds); */
		} else {
			Logger::warn() << "Initializing frame \"" << frame.name << "\"" << std::endl;

			// Frame (name) text
			gui::Text *text_frame = layer.text_render("default")->text(
				frame.name, tpos,
				{1, 1, 1, 1}, scale
			);

			// Time text
			// TODO: use a format string
			gui::Text *text_time = layer.text_render("default")->text(
				"time:", tpos,
				{1, 1, 1, 1}, scale
			);

			// Parent text
			gui::Text *text_parent = layer.text_render("default")->text(
				"parent:", tpos,
				{1, 1, 1, 1}, scale
			);

			/* Box around next frame
			gui::Rect *frame_box = new gui::Rect(
				{0, 0, 0, 0}, color * 0.5f
			); */

			// Add to the element
			element->children.push_back(gui::Element(text_frame));
			element->children.push_back(gui::Element(text_time));
			element->children.push_back(gui::Element(text_parent));

			// element->children.push_back(gui::Element(frame_box));

			/* Recursively add children
			size_t n = 0;
			for (auto &child : frame.children) {
				// One bounding box for each child
				glm::vec4 bounds = {0, 0, 0, 0};
				gui::Rect *child_box = new gui::Rect(
					bounds, color * 0.5f
				);

				element->children.push_back(gui::Element(child_box));
				update_profiler(child, child_box, frame.time);
				n++;
			}

			std::cout << "frame = \"" << frame.name << "\" element position = " << element->position().x << ", " << element->position().y << std::endl; */
		}
	}

	// Record command buffers
	void record(VkCommandBuffer cbuf, VkFramebuffer fbuf) {
		// Pop frame from the profiler
		if (profiler->size() > 0) {
			auto frame = profiler->pop();
			update_profiler(frame, content);
		}

		// Begin recording
		Vulkan::begin(cbuf);

		layer.render(cbuf, fbuf);

		// End the command buffer
		Vulkan::end(cbuf);
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

		record(command_buffers[image_index], swapchain.framebuffers[image_index]);

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

	void frame() override {
		// Check input
		if (input.is_key_down(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(surface.window, true);
			return;
		}

		// Present the frame
		present();
	}
};

#endif
