#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/common.hpp"
#include "include/ecs.hpp"
#include "include/engine/ecs_panel.hpp"
#include "include/io/event.hpp"
#include "include/layers/font_renderer.hpp"
#include "include/layers/raster.hpp"
#include "include/layers/raytracer.hpp"
#include "include/layers/shape_renderer.hpp"
#include "include/logger.hpp"
#include "include/renderer.hpp"
#include "include/scene.hpp"
#include "include/transform.hpp"
#include "include/types.hpp"
#include "include/ui/button.hpp"
#include "include/ui/slider.hpp"
#include "tinyfiledialogs.h"

using namespace kobra;

// Scene path
std::string scene_path = "scenes/scene.kobra";

// Test app
struct ECSApp : public BaseApp {
	layers::Raster	rasterizer;
	layers::Raytracer raytracer;
	layers::FontRenderer font_renderer;
	layers::ShapeRenderer shape_renderer;
	engine::ECSPanel panel;

	Scene scene;

	// TODO: will later also need a project manager

	ui::Button button;
	Entity camera;

	ui::Slider slider;

	static constexpr glm::vec2 window_size {1920, 1200};
	static constexpr float scene_graph_width = 400;
	static constexpr float component_panel_width = 400;
	static constexpr float project_explorer_height = 300;
	static constexpr glm::vec2 render_min = {scene_graph_width, 0};
	static constexpr glm::vec2 render_max = {
		window_size.x - component_panel_width,
		window_size.y - project_explorer_height
	};

	ECSApp(const vk::raii::PhysicalDevice &phdev, const std::vector <const char *> &extensions)
			: BaseApp(phdev, "ECSApp",
				vk::Extent2D {(uint32_t) window_size.x, (uint32_t) window_size.y},
				extensions, vk::AttachmentLoadOp::eLoad
			),
			rasterizer(get_context(), vk::AttachmentLoadOp::eClear),
			raytracer(get_context(), &sync_queue, vk::AttachmentLoadOp::eClear),
			font_renderer(get_context(), render_pass, "resources/fonts/noto_sans.ttf"),
			shape_renderer(get_context(), render_pass),
			panel(get_context(), scene.ecs, io) {
		scene.load(get_device(), scene_path);
		// raytracer.environment_map(scene.p_environment_map);
		raytracer.environment_map("resources/skies/background_1.jpg");

		// Camera
		camera = scene.ecs.get_entity("Camera");

		// Button
		auto main_handler = [](void *) {
			std::cout << "Button pressed (main handler)" << std::endl;
		};

		auto drag_handler = [](void *user, glm::vec2 dpos) {
			auto *app = static_cast <ECSApp *> (user);
			ui::Button &button = app->button;

			button.shape().min += dpos;
			button.shape().max += dpos;
		};

		ui::Button::Args button_args {
			.min = {100, 100},
			.max = {200, 200},
			.radius = 0.01f,
			.border_width = 0.01f,

			.idle = {1.0f, 0.0f, 0.0f},
			.hover = {0.0f, 1.0f, 0.0f},
			.pressed = {0.0f, 0.0f, 1.0f},

			.on_click = {{nullptr, main_handler}},
			.on_drag = {{this, drag_handler}}
		};

		// TODO: ui namespace and directory
		button = ui::Button(io.mouse_events, button_args);

		// Slider
		ui::Slider::Args slider_args {
			.percent = 0.5f,
			.max = {400, 360},
			.min = {100, 340},
			.name = "Slider",
			.font_renderer = &font_renderer,
			.value_func = [](float x) {
				return log(x);
			}
		};

		slider = ui::Slider(io.mouse_events, slider_args);

		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);

		scene.ecs.info <Mesh> ();
	}

	int mode = 0;	// 0 for raster, 1 for raytracer
	bool tab_pressed = false;

	void active_input() {
		float speed = 20.0f * frame_time;

		// Camera movement
		// TODO: remove transform component from camera?
		auto &cam = camera.get <Camera> ();

		glm::vec3 forward = cam.transform.forward();
		glm::vec3 right = cam.transform.right();
		glm::vec3 up = cam.transform.up();

		if (io.input.is_key_down(GLFW_KEY_W))
			cam.transform.move(forward * speed);
		else if (io.input.is_key_down(GLFW_KEY_S))
			cam.transform.move(-forward * speed);

		if (io.input.is_key_down(GLFW_KEY_A))
			cam.transform.move(-right * speed);
		else if (io.input.is_key_down(GLFW_KEY_D))
			cam.transform.move(right * speed);

		if (io.input.is_key_down(GLFW_KEY_E))
			cam.transform.move(up * speed);
		else if (io.input.is_key_down(GLFW_KEY_Q))
			cam.transform.move(-up * speed);

		// Switch mode on tab
		if (io.input.is_key_down(GLFW_KEY_TAB)) {
			if (!tab_pressed) {
				tab_pressed = true;
				mode = (mode + 1) % 2;
			}
		} else {
			tab_pressed = false;
		}
	}

	float fps = 0;
	float time = 0;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		if (frame_time > 0)
			fps = (fps + 1.0f/frame_time) / 2.0f;

		std::vector <ui::Text> texts {
			ui::Text {
				.text = common::sprintf("%.2f fps", fps),
				.anchor = {10, 10},
				.size = 1.0f
			},
		};

		for (auto &s : slider.texts())
			texts.push_back(s);

		std::vector <ui::Rect> rects {
			button.shape(),
		};

		for (auto &r : slider.shapes())
			rects.push_back(r);

		time += frame_time;

		// Input
		active_input();

		// Begin command buffer
		cmd.begin({});

		if (mode == 1)
			raytracer.render(cmd, framebuffer, scene.ecs, {render_min, render_max});
		else
			rasterizer.render(cmd, framebuffer, scene.ecs, {render_min, render_max});

		// Start render pass
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

		font_renderer.render(cmd, texts);
		shape_renderer.render(cmd, rects);

		// TODO: ui layer  will have a push interface each frame

		cmd.endRenderPass();

		cmd.end();

		// V-sync
		sync_queue.push([]() {});
	}

	void terminate() override {
		if (io.input.is_key_down(GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(window.handle, true);
	}

	// Mouse callback
	static void mouse_callback(void *us, const io::MouseEvent &event) {
		static const int pan_button = GLFW_MOUSE_BUTTON_MIDDLE;
		static const int alt_pan_button = GLFW_MOUSE_BUTTON_LEFT;
		static const int select_button = GLFW_MOUSE_BUTTON_LEFT;

		static const float sensitivity = 0.001f;

		static bool first_movement = true;
		static bool dragging = false;
		static bool dragging_select = false;
		static bool gizmo_dragging = false;

		static float px = 0.0f;
		static float py = 0.0f;

		static glm::vec2 previous_dir {0.0f, 0.0f};

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <ECSApp *> (us);
		auto &cam = app.camera.get <Camera> ();

		// Deltas and directions
		float dx = event.xpos - px;
		float dy = event.ypos - py;
		glm::vec2 dir {dx, dy};

		// Dragging only with the drag button
		// TODO: alt left dragging as ewll
		bool is_drag_button = (event.button == pan_button);
		if (event.action == GLFW_PRESS && is_drag_button)
			dragging = true;
		else if (event.action == GLFW_RELEASE && is_drag_button)
			dragging = false;

		// Pan only when draggign
		if (dragging) {
			yaw -= dx * sensitivity;
			pitch -= dy * sensitivity;

			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;

			cam.transform.rotation.x = pitch;
			cam.transform.rotation.y = yaw;
		}

		// Update previous position
		px = event.xpos;
		py = event.ypos;

		previous_dir = dir;
	}
};

int main()
{
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (FIRST)
	auto phdev = pick_physical_device(predicate);

	/* auto camera = Camera {
		Transform { {2, 2, 6}, {-0.1, 0.3, 0} },
		Tunings { 45.0f, 800, 800 }
	}; */

	// Create the app and run it
	ECSApp app(phdev, extensions);
	// RTApp app(phdev, extensions);
	// engine::RTCapture app(phdev, {1000, 1000}, extensions, scene_path, camera);

	// Run the app
	app.run();
}
