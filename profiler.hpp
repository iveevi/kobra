#ifndef PROFILER_APPLICATION_H_
#define PROFILER_APPLICATION_H_

#include "include/profiler.hpp"
#include "include/gui/gui.hpp"
#include "include/gui/text.hpp"
#include "include/gui/area.hpp"
#include "include/gui/button.hpp"
#include "include/gui/layer.hpp"

using namespace kobra;

// Profiler application
class ProfilerApplication : public kobra::App {
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
			context.phdev,
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
				window.coordinates(0, 0),
				window.coordinates(100, 100),
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
			button,
		};

		layer.add(elements);

		make_window();
	}

	// Element that contains the profiler info
	gui::Rect *wborder = nullptr;
	gui::Rect *content = nullptr;

	void make_window() {
		gui::Text *t = layer.text_render("default")->text(
			"My window",
			window.coordinates(300, 50),
			{1, 1, 1, 1}
		);

		wborder = new gui::Rect(
			t->bounding_box(),
			{0.5, 0.5, 0.5}
		);

		content = new gui::Rect(
			window.coordinates(300, 100),
			window.coordinates(100, 100),
			{0.8, 0.8, 0.8}
		);

		wborder->children.push_back(gui::Element(t));
		wborder->children.push_back(gui::Element(content));
		layer.add(wborder);
	}

	// Update the profiler
	// TODO: implement as a nesting of GUI windows
	// TODO: fix bobbing
	void update_profiler(const Profiler::Frame &frame, gui::Rect *element, float parent = -1.0f) {
		// TODO: window method -> pixels
		static float padding = 0.01f;
		static float scale = 0.6f;
		static const int boxoff = 3;

		// Get children
		std::vector <gui::Element> &children = element->children;

		if (children.empty()) {
			// If there are no children, create them
			gui::Text *name = layer.text_render("default")->text(
				frame.name,
				window.coordinates(0, 0),
				{1, 1, 1, 1}, scale * 1.25
			);

			gui::Text *time = layer.text_render("default")->text(
				"time: " + std::to_string(frame.time),
				window.coordinates(0, 0),
				{1, 1, 1, 1}, scale
			);

			gui::Text *parent = layer.text_render("default")->text(
				"parent:",
				window.coordinates(0, 0),
				{1, 1, 1, 1}, scale
			);

			// Add children
			element->children.push_back(gui::Element(name));
			element->children.push_back(gui::Element(time));
			element->children.push_back(gui::Element(parent));

			// Add border boxes for each child
			std::vector <Profiler::Frame> fchildren = frame.children;
			for (size_t i = 0; i < fchildren.size(); i++) {
				gui::Rect *border = new gui::Rect(
					glm::vec4 {0, 0, 0, 0},
					{0.5, 0.5, 0.5}
				);

				element->children.push_back(gui::Element(border));

				// Then process the child frame
				update_profiler(fchildren[i], border, frame.time);
			}
		} else {
			// Get position
			glm::vec2 pos = element->position() + glm::vec2 {padding};

			// Update children
			gui::Text *name = (gui::Text *) children[0].get();
			name->str = frame.name;

			gui::Text *time = (gui::Text *) children[1].get();
			time->str = common::sprintf("time:    %.2f ms", frame.time/1000.0f);

			gui::Text *par = (gui::Text *) children[2].get();
			par->str = "parent: NaN";
			if (parent > 0.0f) {
				float perc = 100 * frame.time / parent;
				par->str = common::sprintf("parent: %.2f%%", perc);
			}

			// Set element positions
			name->pos = pos;
			pos += glm::vec2 {0, name->height() + 0.01f};
			time->pos = pos;
			pos += glm::vec2 {0, time->height() + 0.01f};
			par->pos = pos;

			// Refresh elements
			name->refresh();
			time->refresh();
			par->refresh();

			// Update children
			glm::vec2 offset = pos + glm::vec2 {0.05, 1.5f * par->height()};

			std::vector <Profiler::Frame> fchildren = frame.children;
			for (size_t i = 0; i < fchildren.size(); i++) {
				// Get child element
				gui::Rect *border = (gui::Rect *) children[i + boxoff].get();

				// Update position
				border->set_bounds(glm::vec4 {offset, offset});

				// Update child
				update_profiler(fchildren[i], border, frame.time);

				// Get bbox of children
				glm::vec4 bbox = border->bounding_box();

				// Update offset
				offset.y = bbox.w + 0.05;
			}

			// Get bbox of children and pad it
			glm::vec4 bbox = gui::get_bounding_box(children);
			bbox += padding * glm::vec4 {-1, -1, 1, 1};
			element->set_bounds(bbox);

			// Recalculate child box bounds
			for (size_t i = boxoff; i < children.size(); i++) {
				gui::Rect *border = (gui::Rect *) children[i].get();
				glm::vec4 cbbox = border->bounding_box();
				cbbox.z = bbox.z - padding;
				border->set_bounds(cbbox);
			}
		}
	}

	// Record command buffers
	void record(VkCommandBuffer cbuf, VkFramebuffer fbuf) {
		profiler->frame("Frame window");

		// Pop frame from the profiler
		if (profiler->size() > 0) {
			auto frame = profiler->pop();
			update_profiler(frame, content);
			glm::vec4 bbox = gui::get_bounding_box(wborder->children);
			bbox += 0.025f * glm::vec4 {-1, -1, 1, 1};
			wborder->set_bounds(bbox);
		}

		profiler->end();

		// Begin recording
		profiler->frame("Command buffer");
		Vulkan::begin(cbuf);

		layer.render(cbuf, fbuf);

		// End the command buffer
		Vulkan::end(cbuf);
		profiler->end();
	}

	// Present frame
	void present() {
		profiler->frame("Present");

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

		profiler->frame("Record");
		record(command_buffers[image_index], swapchain.framebuffers[image_index]);
		profiler->end();

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

		profiler->frame("Submit");
		result = vkQueueSubmit(
			context.device.graphics_queue, 1, &submit_info,
			in_flight_fences[frame_index]
		);
		profiler->end();

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

		profiler->end();
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
