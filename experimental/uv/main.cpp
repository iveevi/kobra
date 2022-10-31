// Engine headers
#include "../../include/app.hpp"
#include "../../include/capture.hpp"
#include "../../include/core/interpolation.hpp"
#include "../../include/layers/font_renderer.hpp"
#include "../../include/layers/hybrid_tracer.cuh"
#include "../../include/layers/optix_tracer.cuh"
#include "../../include/layers/forward_renderer.hpp"
#include "../../include/layers/wadjet.cuh"
#include "../../include/optix/options.cuh"
#include "../../include/optix/parameters.cuh"
#include "../../include/scene.hpp"

// TODO: do base app without inheritance (simple struct..., app and baseapp not
// related)
struct MotionCapture : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	kobra::layers::Wadjet tracer;
	kobra::layers::FontRenderer font_renderer;
	kobra::layers::ForwardRenderer forward_renderer;
	
	MotionCapture(const vk::raii::PhysicalDevice &phdev,
			const std::vector <const char *> &extensions,
			const std::string &scene_path)
			: BaseApp(phdev, "MotionCapture",
				vk::Extent2D {1000, 1000},
				extensions, vk::AttachmentLoadOp::eLoad
			),
			font_renderer(get_context(),
				render_pass,
				"resources/fonts/noto_sans.ttf"
			) {
		// Load scene and camera
		scene.load(get_device(), scene_path);
		camera = scene.ecs.get_entity("Camera");

		// Setup Wadjet tracer
		KOBRA_LOG_FILE(kobra::Log::INFO) << "Hybrid tracer setup\n";
		tracer = kobra::layers::Wadjet::make(get_context());
		kobra::layers::set_envmap(tracer, "resources/skies/background_1.jpg");

		// Setup forward renderer
		forward_renderer = kobra::layers::ForwardRenderer::make(get_context());

		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);
		io.keyboard_events.subscribe(keyboard_callback, this);
	}

	float time = 0.0f;
	unsigned int mode = kobra::optix::eRegular;

	std::queue <bool> events;
	std::mutex events_mutex;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();
		
		// Interpolate camera position
		/* glm::vec3 pos = kobra::core::piecewise_linear(camera_pos_seq, time);
		glm::vec3 rot = kobra::core::piecewise_linear(camera_rot_seq, time);

		transform.position = pos;
		transform.rotation = rot; */

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
			/* kobra::layers::compute(tracer,
				scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				mode, accumulate
			);

			kobra::layers::render(tracer, cmd, framebuffer);
			// kobra::layers::capture(tracer, frame); */
			
			kobra::layers::render(forward_renderer,
				scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				cmd, framebuffer
			);

			// TODO: UI layer...
			// Text to render
			kobra::ui::Text t_fps(
				kobra::common::sprintf("%.2f fps", 1.0f/frame_time),
				{5, 5}, glm::vec3 {1, 0.6, 0.6}, 0.7f
			);

			kobra::ui::Text t_samples(
				kobra::common::sprintf("%d samples", tracer.launch_params.samples),
				{0, 5}, glm::vec3 {1, 0.6, 0.6}, 0.7f
			);

			kobra::ui::Text t_mode(
				kobra::common::sprintf("Mode: %s", kobra::optix::str_modes[mode]),
				{5, 45}, glm::vec3 {1, 0.6, 0.6}, 0.5f
			);

			glm::vec2 size = font_renderer.size(t_samples);
			t_samples.anchor.x = 1000 - size.x - 5;
		
			// Start render pass
			// TODO: Make this a function
			std::array <vk::ClearValue, 2> clear_values = {
				vk::ClearValue {
					vk::ClearColorValue {
						std::array <float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
					}
				},
				vk::ClearValue {
					vk::ClearDepthStencilValue {
						1.0f, 0
					}
				}
			};

			cmd.beginRenderPass(
				vk::RenderPassBeginInfo {
					*render_pass,
					*framebuffer,
					vk::Rect2D {
						vk::Offset2D {0, 0},
						extent,
					},
					static_cast <uint32_t> (clear_values.size()),
					clear_values.data()
				},
				vk::SubpassContents::eInline
			);

			font_renderer.render(cmd, {t_fps});

			cmd.endRenderPass();
		cmd.end();

#ifdef RECORD

		// Write the frame to the video
		cv::Mat mat(1000, 1000, CV_8UC4, frame.data());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
		capture.write(mat);

		if (time > camera_pos.size() - 1) {
			capture.release();
			terminate_now();
		}

#endif

		// Update time (fixed)
		time += 1/60.0f;
	}
	
	// Mouse callback
	static void mouse_callback(void *us, const kobra::io::MouseEvent &event) {
		static const int pan_button = GLFW_MOUSE_BUTTON_MIDDLE;

		static const float sensitivity = 0.001f;

		static float px = 0.0f;
		static float py = 0.0f;

		static glm::vec2 previous_dir {0.0f, 0.0f};

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <MotionCapture *> (us);
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
		auto &app = *static_cast <MotionCapture *> (us);
		auto &transform = app.camera.get <kobra::Transform> ();

		// M to switch between modes
		if (event.key == GLFW_KEY_M && event.action == GLFW_PRESS) {
			app.mode = (app.mode + 1) % kobra::optix::eCount;
			
			// Add to event queue
			app.events_mutex.lock();
			app.events.push(true);
			app.events_mutex.unlock();
		}

		// Numbers 1-3 for the same function
		if (event.key >= GLFW_KEY_1 && event.key <= GLFW_KEY_3 && event.action == GLFW_PRESS) {
			app.mode = event.key - GLFW_KEY_1;
			
			// Add to event queue
			app.events_mutex.lock();
			app.events.push(true);
			app.events_mutex.unlock();
		}

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
	MotionCapture app(phdev, {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	}, scene_path);

	app.run();
}