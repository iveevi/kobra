// Engine headers
#include "../../include/app.hpp"
#include "../../include/capture.hpp"
#include "../../include/common.hpp"
#include "../../include/core/interpolation.hpp"
#include "../../include/layers/font_renderer.hpp"
#include "../../include/layers/forward_renderer.hpp"
#include "../../include/layers/imgui.hpp"
#include "../../include/scene.hpp"
#include "../../include/shader_program.hpp"
#include "../../include/ui/framerate.hpp"

#define CWD KOBRA_DIR "/experimental/uv"

struct UVMapper : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	kobra::layers::ForwardRenderer forward_renderer;
	kobra::layers::ImGUI imgui;
	
	UVMapper(const vk::raii::PhysicalDevice &phdev,
			const std::vector <const char *> &extensions,
			const std::string &scene_path)
			: BaseApp(phdev, "UVMapper",
				vk::Extent2D {1000, 1000},
				extensions, vk::AttachmentLoadOp::eLoad
			) {
		// Load scene and camera
		scene.load(get_device(), scene_path);
		camera = scene.ecs.get_entity("Camera");

		// Setup forward renderer
		forward_renderer = kobra::layers::ForwardRenderer(get_context());

		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);
		io.keyboard_events.subscribe(keyboard_callback, this);

		// Initialize ImGUI
		// TODO: ui/core method (initialize)
		ImGui::CreateContext();
		ImPlot::CreateContext();

		ImGui_ImplGlfw_InitForVulkan(window.handle, true);

		imgui = kobra::layers::ImGUI(get_context(), window, graphics_queue);
		imgui.set_font(KOBRA_DIR "/resources/fonts/NotoSans.ttf", 30);

		imgui.attach(std::make_shared <kobra::ui::FramerateAttachment> ());
	}

	float time = 0.0f;

	std::queue <bool> events;
	std::mutex events_mutex;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();

		float speed = 20.0f * frame_time;
		
		glm::vec3 forward = transform.forward();
		glm::vec3 right = transform.right();
		glm::vec3 up = transform.up();

		bool accumulate = true;

		if (io.input.is_key_down(GLFW_KEY_W)) {
			transform.move(forward * speed);
			accumulate = false;
		} else if (io.input.is_key_down(GLFW_KEY_S)) {
			transform.move(-forward * speed);
			accumulate = false;
		}

		if (io.input.is_key_down(GLFW_KEY_A)) {
			transform.move(-right * speed);
			accumulate = false;
		} else if (io.input.is_key_down(GLFW_KEY_D)) {
			transform.move(right * speed);
			accumulate = false;
		}

		if (io.input.is_key_down(GLFW_KEY_E)) {
			transform.move(up * speed);
			accumulate = false;
		} else if (io.input.is_key_down(GLFW_KEY_Q)) {
			transform.move(-up * speed);
			accumulate = false;
		}

		// Also check our events
		events_mutex.lock();
		if (!events.empty())
			accumulate = false; // Means that camera direction
					    // changed

		events = std::queue <bool> (); // Clear events
		events_mutex.unlock();

		// Now trace and render
		cmd.begin({});
			forward_renderer.render(
				scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				cmd, framebuffer
			);

			imgui.render(cmd, framebuffer, extent);
		cmd.end();

		// Update time (fixed)
		time += 1/60.0f;
	}
	
	// Mouse callback
	static void mouse_callback(void *us, const kobra::io::MouseEvent &event) {
		static const int pan_button = GLFW_MOUSE_BUTTON_RIGHT;

		static const float sensitivity = 0.001f;

		static float px = 0.0f;
		static float py = 0.0f;

		static glm::vec2 previous_dir {0.0f, 0.0f};

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <UVMapper *> (us);
		auto &transform = app.camera.get <kobra::Transform> ();

		// Deltas and directions
		float dx = event.xpos - px;
		float dy = event.ypos - py;
		glm::vec2 dir {dx, dy};
		
		// Check if panning
		static bool dragging = false;
		static bool alt_dragging = false;

		bool is_drag_button = (event.button == pan_button);
		if (event.action == GLFW_PRESS && is_drag_button)
			dragging = true;
		else if (event.action == GLFW_RELEASE && is_drag_button)
			dragging = false;

		bool is_alt_down = app.io.input.is_key_down(GLFW_KEY_LEFT_ALT);
		if (!alt_dragging && is_alt_down)
			alt_dragging = true;
		else if (alt_dragging && !is_alt_down)
			alt_dragging = false;

		// Pan only when dragging
		if (dragging || alt_dragging) {
			yaw -= dx * sensitivity;
			pitch -= dy * sensitivity;

			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;

			transform.rotation.x = pitch;
			transform.rotation.y = yaw;

			// Add to event queue
			app.events_mutex.lock();
			app.events.push(true);
			app.events_mutex.unlock();
		}

		// Update previous position
		px = event.xpos;
		py = event.ypos;

		previous_dir = dir;
	}
	
	// Keyboard callback
	static void keyboard_callback(void *us, const kobra::io::KeyboardEvent &event) {
		auto &app = *static_cast <UVMapper *> (us);
		auto &transform = app.camera.get <kobra::Transform> ();

		// I for info
		if (event.key == GLFW_KEY_I && event.action == GLFW_PRESS) {
			std::cout << "Camera transform:\n";
			std::cout << "\tPosition: " << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << "\n";
			std::cout << "\tRotation: " << transform.rotation.x << ", " << transform.rotation.y << ", " << transform.rotation.z << "\n";
		}
	}

};

int main()
{
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (FIRST)
	auto phdev = kobra::pick_physical_device(predicate);

	std::cout << "Extensions:" << std::endl;
	for (auto str : extensions)
		std::cout << "\t" << str << std::endl;

	const std::string scene_path = "/home/venki/models/sponza.kobra";

	UVMapper app {
		phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		},
		scene_path
	};

	app.run();
}
