#include "../include/app.hpp"
#include "../include/backend.hpp"
#include "../include/layers/forward_renderer.hpp"
#include "../include/layers/ui.hpp"
#include "../include/project.hpp"
#include "../include/scene.hpp"

int main()
{
	// Load Vulkan physical device
	auto predicate = [](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev,  {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		});
	};

	vk::raii::PhysicalDevice phdev = kobra::pick_physical_device(predicate);

	struct Editor : public kobra::BaseApp {
		kobra::Scene scene;
		kobra::Entity camera;
		kobra::layers::ForwardRenderer forward_renderer;

		Editor(const vk::raii::PhysicalDevice &phdev,
				const std::vector <const char *> extensions)
				: kobra::BaseApp {
					phdev, "Stress Test",
					vk::Extent2D {1500, 1000},
					extensions
				} {
			// Setup forward renderer
			forward_renderer = kobra::layers::ForwardRenderer(get_context());

			kobra::Project project = kobra::Project::load(".kobra/project");
			scene.load(get_context(), project.scene);
			
			camera = scene.ecs.get_entity("Camera");
			camera.get <kobra::Camera> ().aspect = 1.5f;
			
			// Mouse callbacks
			io.mouse_events.subscribe(
				[&](void *us, const kobra::io::MouseEvent &event) {
					static const int pan_button = GLFW_MOUSE_BUTTON_RIGHT;

					static const float sensitivity = 0.001f;

					static float px = 0.0f;
					static float py = 0.0f;

					static float yaw = 0.0f;
					static float pitch = 0.0f;

					// Deltas and directions
					float dx = event.xpos - px;
					float dy = event.ypos - py;
					
					// Check if panning
					static bool dragging = false;
					static bool alt_dragging = false;

					bool is_drag_button = (event.button == pan_button);
					if (event.action == GLFW_PRESS && is_drag_button)
						dragging = true;
					else if (event.action == GLFW_RELEASE && is_drag_button)
						dragging = false;

					bool is_alt_down = io.input->is_key_down(GLFW_KEY_LEFT_ALT);
					if (!alt_dragging && is_alt_down)
						alt_dragging = true;
					else if (alt_dragging && !is_alt_down)
						alt_dragging = false;

					// Pan only when dragging
					if (dragging | alt_dragging) {
						yaw -= dx * sensitivity;
						pitch -= dy * sensitivity;

						if (pitch > 89.0f)
							pitch = 89.0f;
						if (pitch < -89.0f)
							pitch = -89.0f;

						kobra::Transform &transform = camera.get <kobra::Transform> ();
						transform.rotation.x = pitch;
						transform.rotation.y = yaw;
					}

					// Update previous position
					px = event.xpos;
					py = event.ypos;
				},
				this
			);
		}

		void record(const vk::raii::CommandBuffer &cmd,
				const vk::raii::Framebuffer &framebuffer) override {
			// Camera movement
			auto &transform = camera.get <kobra::Transform> ();
		
			float speed = 20.0f * frame_time;
		
			glm::vec3 forward = transform.forward();
			glm::vec3 right = transform.right();
			glm::vec3 up = transform.up();

			if (io.input->is_key_down(GLFW_KEY_W))
				transform.move(forward * speed);
			else if (io.input->is_key_down(GLFW_KEY_S))
				transform.move(-forward * speed);

			if (io.input->is_key_down(GLFW_KEY_A))
				transform.move(-right * speed);
			else if (io.input->is_key_down(GLFW_KEY_D))
				transform.move(right * speed);

			if (io.input->is_key_down(GLFW_KEY_E))
				transform.move(up * speed);
			else if (io.input->is_key_down(GLFW_KEY_Q))
				transform.move(-up * speed);

			cmd.begin({});
				forward_renderer.render(
					scene.ecs,
					camera.get <kobra::Camera> (),
					camera.get <kobra::Transform> (),
					cmd, framebuffer, extent
				);
			cmd.end();
		}

		void resize(const vk::Extent2D &extent) override {
			camera.get <kobra::Camera> ().aspect = extent.width / (float) extent.height;
		}
	};

	Editor editor {
		phdev,
		{VK_KHR_SWAPCHAIN_EXTENSION_NAME},
	};

	editor.run();
}
